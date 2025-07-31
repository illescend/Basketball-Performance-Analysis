import math
import os

from matplotlib import pyplot as plt

os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"  # Try to disable Media Foundation
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "100"  # Try to prioritize FFMPEG
import numpy as np
import cv2
from ultralytics import YOLO
from src.utils import detection_utils as utils
from src.utils import trajectory_utils as tu

import torch
import torchvision.transforms as transforms
from src.models.release_model.model_def import ReleaseRelationNet # Import your model definition

# --- Global Constants and Configuration HERO12 ---
# K = np.array([[1134.4909, 0.0, 968.5293], [0.0, 1133.2194, 553.0426], [0.0, 0.0, 1.0]])
# D = np.array([0.43466, -0.63731, 2.29794, -2.09685])  # Fisheye

# --- Global Constant and Config HERO7
K = np.array([[1006.894800376467,0.0,964.3578313551827],[0.0,994.7402914201482,537.3734258659648],[0.0,0.0, 1.0]])
D = np.array([0.36097005855002745,-0.013703354643132513,0.45553079194926527,-0.32891315019398665])  # Fisheye

fx = K[0, 0];
fy = K[1, 1];
cx = K[0, 2];
cy = K[1, 2]



SHOOTING_WRIST_KP_INDEX = utils.SHOOTING_WRIST_KP_INDEX
MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION = utils.MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
WRIST_BALL_PROXIMITY_MARGIN = utils.WRIST_BALL_PROXIMITY_MARGIN
CONFIRMATION_FRAMES_FOR_RELEASE = utils.CONFIRMATION_FRAMES_FOR_RELEASE

SHOOTING_ARM_WRIST_IDX = utils.RIGHT_WRIST
SHOOTING_ARM_ELBOW_IDX = utils.RIGHT_ELBOW
SHOOTING_ARM_SHOULDER_IDX = utils.RIGHT_SHOULDER

SAVE_RELEASE_FRAMES = True
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# RELEASE_FRAME_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "ccs", "release_frames_kf_compare") # Original relative path
#RELEASE_FRAME_OUTPUT_DIR = "outputs/release_frames_kf_compare"  # Effective path
RELEASE_FRAME_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "trajectory_fitting", "wcs_release_frames")  # Changed output dir slightly
os.makedirs(RELEASE_FRAME_OUTPUT_DIR, exist_ok=True)

PRE_RELEASE_BALL_CONF = 0.35
POST_RELEASE_BALL_CONF = 0.25
MAX_PIXEL_DISPLACEMENT_POST_RELEASE = 100
FRAMES_FOR_NAIVE_ESTIMATION = 20
BASKETBALL_REAL_DIAMETER_M = 0.24
TARGET_BALL_CLASS_NAME = "ball"

#3D World Points
WCS_FT_LINE_TO_RIM_CENTER_X = 4.191  # m (X-coord of rim center)
WCS_RIM_HEIGHT_Y = 3.048           # m (Y-coord of rim)
WCS_RIM_RADIUS = 0.4572 / 2        # m
WCS_BACKBOARD_OFFSET_FROM_RIM_CENTER = WCS_RIM_RADIUS + 0.15 # m (backboard is behind rim center)
WCS_BACKBOARD_X = WCS_FT_LINE_TO_RIM_CENTER_X + WCS_BACKBOARD_OFFSET_FROM_RIM_CENTER # X-coord of backboard face
WCS_BB_INNER_RECT_WIDTH = 0.61     # m
WCS_BB_INNER_RECT_HEIGHT = 0.45    # m
WCS_BB_INNER_RECT_BOTTOM_Y = WCS_RIM_HEIGHT_Y - WCS_BB_INNER_RECT_HEIGHT # Y-coord of inner rectangle bottom
WCS_BB_INNER_RECT_TOP_Y = WCS_RIM_HEIGHT_Y + (WCS_BB_INNER_RECT_HEIGHT - 0.05)

MANUAL_RELEASE_FRAME_OVERRIDE = None #111 #176  # 180 #111 for freethrow_2
MANUAL_RELEASE_SHOT_NUMBER_TARGET = 1  # Which shot number to apply manual override
MAX_CONSECUTIVE_MISSES_KF = 5
KF_MEASUREMENT_NOISE_SCALE = 0.001 #Confidence in pixel detection
KF_PROCESS_NOISE_SCALE = 300  # Increased from 0.1 for more dynamic movement
NOISE_ACCELERATION_VARIANCE = 1536

UNDISTORT_POINTS_AFTER_DETECTION = True
DETECT_ON_FULLY_UNDISTORTED_FRAME = True  # Set to False as per "New" script

# --- NEW: Model and Inference Configuration ---
MODEL_PATH = os.path.join(BASE_DIR, "src", "weights", "best_release_model.pth") # Changed output dir slightly
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 96 # Must match training
INFERENCE_WINDOW_BEFORE = 10 # How many frames before heuristic to check
INFERENCE_WINDOW_AFTER = 5   # How many frames after heuristic to check
SHOT_DURATION_FRAMES = 85
SHOT_COOLDOWN_PERIOD = 50 # Frames

# --- Filtering Parameters (can be the same as data_collection) ---
MIN_BALL_HEIGHT_FOR_SHOT_GATHER = 0.45
MIN_WRIST_HEIGHT_FOR_SHOT_GATHER = 0.55

# --- Global variable for the loaded model ---
release_model = None

#Plot variables
# Court dimensions for plotting (should align with WCS definitions)
# These are used by the plotting function later.
# WCS_FT_LINE_TO_RIM_CENTER_X, WCS_RIM_HEIGHT_Y are already defined for solvePnP.
# We might need a few more for detailed backboard plotting if desired.
PLOT_RIM_DIAMETER = 0.4572
PLOT_BACKBOARD_X = WCS_BACKBOARD_X # Use the same as solvePnP
PLOT_BACKBOARD_HEIGHT = 1.0668 # Standard backboard height
# Assuming inner rectangle bottom is at rim height for plotting
PLOT_BACKBOARD_RECT_BOTTOM_Y = WCS_RIM_HEIGHT_Y
PLOT_BACKBOARD_RECT_TOP_Y = WCS_RIM_HEIGHT_Y + WCS_BB_INNER_RECT_HEIGHT # From solvePnP setup
# If plotting the full backboard, its bottom might be lower
PLOT_BACKBOARD_BOTTOM_Y = WCS_RIM_HEIGHT_Y - (WCS_BB_INNER_RECT_HEIGHT/2) - (PLOT_BACKBOARD_HEIGHT - WCS_BB_INNER_RECT_HEIGHT)/2

# --- New WCS Globals ---
R_wcs_to_ccs = None
tvec_wcs_to_ccs = None
wcs_setup_done = False

# --- Trajectory Fitting Globals
fitted_params_dict = None # Will store the dictionary returned by the fitting function
fitting_performed_for_shot = 0 # To track which shot was last fitted


def _init_release_state(u_rel, v_rel, release_frame_num, bbox_at_release):
    """Internal helper to initialize KF and reset current shot's trajectories."""
    global Zc_at_release, kf_initialized, estimated_speed_mps_wcs, estimated_angle_deg_elev_wcs, estimated_angle_deg_azim_wcs

    # Reset current trajectory lists and estimates for the new shot
    trajectory_points_raw_pixels_undistorted.clear()
    trajectory_points_kf_pixels_undistorted.clear()
    trajectory_points_ccs.clear()
    trajectory_points_wcs.clear()

    estimated_speed_mps_wcs, estimated_angle_deg_elev_wcs, estimated_angle_deg_azim_wcs = 0.0, 0.0, 0.0

    app_diam_px_rel = ((bbox_at_release[2] - bbox_at_release[0]) + (bbox_at_release[3] - bbox_at_release[1])) / 2.0
    Zc_at_release = (BASKETBALL_REAL_DIAMETER_M * (fx + fy) / 2.0) / app_diam_px_rel if app_diam_px_rel > 0 else 1.0
    Zc_at_release = 3
    print(Zc_at_release)

    kf.statePost = np.array([u_rel, v_rel, 0, 0], dtype=np.float32).reshape(-1, 1)
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 50.0
    kf_initialized = True

    xn_rel = (u_rel - cx) / fx;
    yn_rel = (v_rel - cy) / fy
    Pc_rel_ccs = np.array([xn_rel * Zc_at_release, yn_rel * Zc_at_release, Zc_at_release])
    Pc_rel_wcs = ccs_to_wcs(Pc_rel_ccs)

    trajectory_points_kf_pixels_undistorted.append(
        {'frame': release_frame_num, 'px_coord': (u_rel, v_rel), 'type': 'kf_model_init'})
    trajectory_points_ccs.append({'frame': release_frame_num, 'Pc_ccs': Pc_rel_ccs, 'px_coord_type': 'kf_model_init'})
    if Pc_rel_wcs is not None:
        trajectory_points_wcs.append(
            {'frame': release_frame_num, 'Pc_wcs': Pc_rel_wcs, 'px_coord_type': 'kf_model_init'})


def initiate_shot_tracking(refined_frame_num, frame_buffer, ball_model, pose_model):
    """
    Finds frame, runs final detection, creates a new video clip (with annotated pre-release),
    and initializes tracking state.
    """
    global shot_count, release_detected_in_shot, release_frame_info_dict, shot_active_timer
    global shot_video_writer

    # Find the data for the exact refined release frame from the buffer
    release_frame_data = None
    for data in frame_buffer:
        if data['frame_num'] == refined_frame_num:
            release_frame_data = data
            break

    if release_frame_data is None:
        print(f"Error: Could not find model-refined frame {refined_frame_num} in buffer.")
        return False

    # Use the detections already stored for the release frame
    final_ball_center = release_frame_data['ball_center']
    final_ball_bbox = release_frame_data['ball_bbox']

    if final_ball_center is not None and final_ball_bbox is not None:
        shot_count += 1
        release_detected_in_shot = True
        shot_active_timer = SHOT_DURATION_FRAMES
        release_frame_info_dict = {
            "frame_no": refined_frame_num,
            "ball_bbox_detection": final_ball_bbox,
        }

        # --- NEW: VIDEO CLIP CREATION WITH NEW FOLDER STRUCTURE ---
        shot_dir = os.path.join(session_output_dir, f"shot_{shot_count:03d}")
        os.makedirs(shot_dir, exist_ok=True)

        # Use a more descriptive name for the clip
        clip_output_path = os.path.join(shot_dir, f"shot_{shot_count:03d}_clip.avi")
        shot_video_writer = cv2.VideoWriter(clip_output_path, cv2.VideoWriter_fourcc(*'MJPG'), video_fps_g,
                                            (frame_width_g, frame_height_g))

        if not shot_video_writer.isOpened():
            print(f"!!! ERROR: Could not create video writer for shot clip: {clip_output_path}")
            release_detected_in_shot = False
            return False

        # --- WRITE ANNOTATED PRE-RELEASE FRAMES ---
        start_frame_for_clip = refined_frame_num - INFERENCE_WINDOW_BEFORE
        for data in frame_buffer:
            f_num = data['frame_num']
            if start_frame_for_clip <= f_num < refined_frame_num:
                # Reconstruct the annotated frame for this past frame
                pre_release_img = data['image']
                pre_release_annotated = preprocess_frame_ccs(pre_release_img, K, D, map1_g, map2_g,
                                                             DETECT_ON_FULLY_UNDISTORTED_FRAME)

                # Use the stored detections to draw annotations
                draw_annotations_on_frame_ccs(
                    pre_release_annotated,
                    f_num,
                    data['ball_bbox'],
                    data['ball_center'],
                    data['wrist_pos'],
                    False,  # is_release_detected is False for these frames
                    None,  # release_info_dict is None
                    [],  # kf_trajectory is empty
                    [],  # raw_trajectory is empty
                    shot_count,  # Show the upcoming shot number
                    0, 0,  # No speed/angle estimates yet
                    data['low_conf_flag'],
                    None  # No kf prediction yet
                )
                shot_video_writer.write(pre_release_annotated)

        # Initialize the state for the actual tracking to begin from the next frame
        _init_release_state(final_ball_center[0], final_ball_center[1], refined_frame_num, final_ball_bbox)
        print(f"\n>>> SHOT {shot_count} INITIATED (Model-Refined Frame: {refined_frame_num}) <<<")
        return True
    return False

def load_release_model(model_path):
    """Loads the trained PyTorch model and sets it to evaluation mode."""
    global release_model
    print(f"Loading release refinement model from: {model_path}")
    try:
        model = ReleaseRelationNet().to(MODEL_DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=MODEL_DEVICE))
        model.eval()  # IMPORTANT: Set model to evaluation mode
        release_model = model
        print("Release refinement model loaded successfully.")
    except Exception as e:
        print(f"!!! ERROR: Could not load release model: {e}")
        release_model = None


def crop_patch(full_frame, center_xy, patch_size):
    """Crops a square patch, handling boundaries by padding."""
    if center_xy is None: return None
    cx, cy = center_xy
    half_size = patch_size // 2
    x1, y1 = cx - half_size, cy - half_size
    x2, y2 = cx + half_size, cy + half_size
    h, w, _ = full_frame.shape
    padded_frame = cv2.copyMakeBorder(full_frame, half_size, half_size, half_size, half_size, cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])
    patch = padded_frame[y1 + half_size: y2 + half_size, x1 + half_size: x2 + half_size]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    return patch


def refine_release_frame_with_model(candidate_frame, frame_buffer):
    """
    Analyzes a window of frames around a candidate and returns the model-predicted release frame.
    This version is updated to use the new frame_buffer structure (list of dicts).
    """
    if release_model is None:
        print("Warning: Release model not loaded. Returning heuristic frame.")
        return candidate_frame

    print(f"--- Running release refinement model around frame {candidate_frame} ---")
    start_frame = candidate_frame - INFERENCE_WINDOW_BEFORE
    end_frame = candidate_frame + INFERENCE_WINDOW_AFTER

    # --- CORRECTED: Build frame_map from the list of dictionaries ---
    # The key is the frame number, the value is the entire data dictionary for that frame.
    frame_map = {data['frame_num']: data for data in frame_buffer}

    # Prepare data for the model
    inference_samples = []
    sorted_frames_in_window = sorted([f_num for f_num in frame_map.keys() if start_frame <= f_num <= end_frame])

    for i, f_num_t in enumerate(sorted_frames_in_window):
        if i == 0: continue  # Need a previous frame

        f_num_t_minus_1 = sorted_frames_in_window[i - 1]

        # Get the full data dictionaries for frame t and t-1
        data_t = frame_map.get(f_num_t)
        data_t_minus_1 = frame_map.get(f_num_t_minus_1)

        if data_t is None or data_t_minus_1 is None:
            continue

        # --- NEW EFFICIENCY: Use stored detections to get patches ---
        # No need to re-run models here.
        img_t = data_t['image']
        ball_bbox_t = data_t['ball_bbox']
        wrist_pos_t = data_t['wrist_pos']

        img_t_minus_1 = data_t_minus_1['image']
        ball_bbox_t_minus_1 = data_t_minus_1['ball_bbox']
        wrist_pos_t_minus_1 = data_t_minus_1['wrist_pos']

        # Check if we have the necessary info for both frames
        if all(item is not None for item in [ball_bbox_t, wrist_pos_t, ball_bbox_t_minus_1, wrist_pos_t_minus_1]):
            ball_center_t = (int((ball_bbox_t[0] + ball_bbox_t[2]) / 2), int((ball_bbox_t[1] + ball_bbox_t[3]) / 2))
            ball_center_t_minus_1 = (int((ball_bbox_t_minus_1[0] + ball_bbox_t_minus_1[2]) / 2),
                                     int((ball_bbox_t_minus_1[1] + ball_bbox_t_minus_1[3]) / 2))

            hand_patch_t = crop_patch(img_t, wrist_pos_t, PATCH_SIZE)
            ball_patch_t = crop_patch(img_t, ball_center_t, PATCH_SIZE)

            hand_patch_t_1 = crop_patch(img_t_minus_1, wrist_pos_t_minus_1, PATCH_SIZE)
            ball_patch_t_1 = crop_patch(img_t_minus_1, ball_center_t_minus_1, PATCH_SIZE)

            if all(p is not None for p in [hand_patch_t, ball_patch_t, hand_patch_t_1, ball_patch_t_1]):
                inference_samples.append({
                    "frame_num": f_num_t,
                    "hand_t": hand_patch_t, "hand_t-1": hand_patch_t_1,
                    "ball_t": ball_patch_t, "ball_t-1": ball_patch_t_1
                })

    if not inference_samples:
        print("Could not generate valid patches for model inference. Returning heuristic frame.")
        return candidate_frame

    # --- Perform inference (this part remains the same) ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    scores = []
    frame_numbers = []
    with torch.no_grad():
        for sample in inference_samples:
            hand_t = transform(cv2.cvtColor(sample['hand_t'], cv2.COLOR_BGR2RGB)).unsqueeze(0).to(MODEL_DEVICE)
            hand_t_1 = transform(cv2.cvtColor(sample['hand_t-1'], cv2.COLOR_BGR2RGB)).unsqueeze(0).to(MODEL_DEVICE)
            ball_t = transform(cv2.cvtColor(sample['ball_t'], cv2.COLOR_BGR2RGB)).unsqueeze(0).to(MODEL_DEVICE)
            ball_t_1 = transform(cv2.cvtColor(sample['ball_t-1'], cv2.COLOR_BGR2RGB)).unsqueeze(0).to(MODEL_DEVICE)

            score = release_model(hand_t, hand_t_1, ball_t, ball_t_1).item()
            scores.append(score)
            frame_numbers.append(sample['frame_num'])

    if not scores:
        print("Model did not produce scores. Returning heuristic frame.")
        return candidate_frame

    scores = np.array(scores)
    # Find the frame with the largest increase in score from the previous frame
    # A sharp positive jump indicates the transition from "not released" to "released"
    if len(scores) > 1:
        score_diffs = np.diff(scores, prepend=scores[0])
        best_frame_index = np.argmax(score_diffs)
    else:  # Only one frame was valid to score
        score_diffs = np.array([scores[0]])
        best_frame_index = 0

    model_refined_frame = frame_numbers[best_frame_index]

    print(f"Model scores: {[f'{f}:{s:.2f}' for f, s in zip(frame_numbers, scores)]}")
    print(f"Score differences: {score_diffs.round(2)}")
    print(f"Heuristic frame: {candidate_frame}, Model refined frame: {model_refined_frame}")

    return model_refined_frame

def setup_wcs(camera_matrix, dist_coeffs_val):
    global R_wcs_to_ccs, tvec_wcs_to_ccs, wcs_setup_done

    # 2D Image Points (picked from the undistorted frame)
    # These points correspond to the user-provided image and red dots / estimated features.
    # Ensure these pixel coordinates are accurate for *your specific frame/video setup*.

    # #Session 1
    # image_points = np.array([
    #     (800, 790),  # P1: Shooter's left foot
    #     (1368, 412),  # P2: Front-left of rim
    #     (1368, 375),  # P3: Backboard top-left of inner rectangle
    #     (1400, 355)  # P4: Backboard top-right of inner rectangle
    # ], dtype=np.float32)

    # #Session 2
    # image_points = np.array([
    #     (827, 827),  # P1: Shooter's left foot
    #     (1672, 310),  # P2: Front-left of rim
    #     (1672, 263),  # P3: Backboard top-left of inner rectangle
    #     (1742, 235)  # P4: Backboard top-right of inner rectangle
    # ], dtype=np.float32)
    #
    # # 3D World Points (defined in WCS: Origin at FT line center, X right, Y towards basket, Z up)
    # world_points = np.array([
    #     [0.0, 0.0, 0.0],  # P1: Shooter's left foot (world_p1)
    #     [WCS_FT_LINE_TO_RIM_CENTER_X, WCS_RIM_HEIGHT_Y, WCS_RIM_RADIUS],  # P2: right of rim (from camera pov) (world_p2_rim)
    #     [WCS_BACKBOARD_X, WCS_BB_INNER_RECT_TOP_Y, -WCS_BB_INNER_RECT_WIDTH/2],  # P3: BB_UL (world_bb_ul)
    #     [WCS_BACKBOARD_X, WCS_BB_INNER_RECT_TOP_Y, WCS_BB_INNER_RECT_WIDTH/2]  # P4: BB_UR (world_bb_ur)
    # ], dtype=np.float32)

    world_points = np.array([
        #[0.0, 0.0, 0.0],  # P1: Origin (back-left corner of the 2x3.3m lane)
        [0.002955, 0.081709, 0.012093],  # P2: left front
        [0.050017, 0.030884, 1.955751],  # P3: left back
        #[3.3, 0.0, 1.0],  # P4: Front-right corner of the lane (at 3.3m length, Z=1)
        [3.297116, 0, 0.046505],  # P5: right front
        [3.110943, 1.562576, 0.639417],  # P6: table front
        [3.140699, 1.546863, 1.433986],  # P7: table back
        [0.65,1.12,0.93], #BALL BATCH 1
        #[0.611321,1.267435,0.985766], #Ball batch 2
        #[0.690223,1.208302,0.937263], #ball batch 3
        #[0.568437,1.239614,1.045756] #ball batch 4
        #[0.776815, 1.314616, 0.94246] #ball batch 5
        #[0.526551,1.339163,0.962354] #ball batch 6
        #[0.435131,1.398297,0.921609] #ball batch 7
        #[0.669286,1.401199,0.952018] #ball batch 8
        #[0.444168,1.478248,0.96978] # ball batch 9
        #[0.4578, 1.521772, 0.977554] #ball batch 10

        [0,0,1] #shooter foot (roughly in line with shot)
    ], dtype=np.float32)

    # --- IMPORTANT: These image_points are placeholders. ---
    # You MUST replace these with the actual 2D pixel coordinates (u, v)
    # manually picked from a single, specific undistorted GoPro frame
    # that clearly shows all the corresponding world_points.
    # The order of these image_points MUST match the order of world_points exactly.

    #Test batch
    # image_points = np.array([
    #     # Example placeholders - REPLACE WITH YOUR ACTUAL PIXEL COORDINATES:
    #     #[x_pixel_for_P1, y_pixel_for_P1],  # P1: (0,0,0)
    #     [79, 1075],  # P2: (0,0,-1)
    #     [507, 817],  # P3: (0,0,1)
    #     #[x_pixel_for_P4, y_pixel_for_P4],  # P4: (3.3,0,-1)
    #     [1815, 1060],  # P6: (3,1.58,-0.4)
    #     [1555, 341],  # P5: (3.3,0,1)
    #     [1411, 389]  # P7: (3,1.58,0.4)
    # ], dtype=np.float32)

    # Batch 1
    image_points = np.array([
        # Example placeholders - REPLACE WITH YOUR ACTUAL PIXEL COORDINATES:
        # [x_pixel_for_P1, y_pixel_for_P1],  # P1: (0,0,0)
        [50,1045],  # P2: (0,0,-1)
        [479,780],  # P3: (0,0,1)
        # [x_pixel_for_P4, y_pixel_for_P4],  # P4: (3.3,0,-1)
        [1761, 1011],  # P6: (3,1.58,-0.4)
        [1523,315],  # P5: (3.3,0,1)
        [1383,361],  # P7: (3,1.58,0.4)
        [570,466], #BALL AT RELEASE
        [296,907] # shooter foot
    ], dtype=np.float32)

    # Batch 2
    # image_points = np.array([
    #     # Example placeholders - REPLACE WITH YOUR ACTUAL PIXEL COORDINATES:
    #     # [x_pixel_for_P1, y_pixel_for_P1],  # P1: (0,0,0)
    #     [43,1061],  # P2: (0,0,-1)
    #     [476,790],  # P3: (0,0,1)
    #     # [x_pixel_for_P4, y_pixel_for_P4],  # P4: (3.3,0,-1)
    #     [1763, 1022],  # P6: (3,1.58,-0.4)
    #     [1521,324],  # P5: (3.3,0,1)
    #     [1380,371],  # P7: (3,1.58,0.4)
    #     [567,414],  # BALL AT RELEASE
    #     [304, 907]  # shooter foot
    # ], dtype=np.float32)

    # Batch 3
    # image_points = np.array([
    #     # Example placeholders - REPLACE WITH YOUR ACTUAL PIXEL COORDINATES:
    #     # [x_pixel_for_P1, y_pixel_for_P1],  # P1: (0,0,0)
    #     [43, 1057],  # P2: (0,0,-1)
    #     [475, 786],  # P3: (0,0,1)
    #     # [x_pixel_for_P4, y_pixel_for_P4],  # P4: (3.3,0,-1)
    #     [1761, 1014],  # P6: (3,1.58,-0.4)
    #     [1516, 318],  # P5: (3.3,0,1)
    #     [1377, 366],  # P7: (3,1.58,0.4)
    #     [527,467],
    #     [352,875]
    # ], dtype=np.float32)

    # Batch 4
    # image_points = np.array([
    #     # Example placeholders - REPLACE WITH YOUR ACTUAL PIXEL COORDINATES:
    #     # [x_pixel_for_P1, y_pixel_for_P1],  # P1: (0,0,0)
    #     [44,1059],  # P2: (0,0,-1)
    #     [476,788],  # P3: (0,0,1)
    #     # [x_pixel_for_P4, y_pixel_for_P4],  # P4: (3.3,0,-1)
    #     [1763, 1021],  # P6: (3,1.58,-0.4)
    #     [1520,324],  # P5: (3.3,0,1)
    #     [1380,372],  # P7: (3,1.58,0.4)
    #     [565, 432],
    #     [326, 899]
    # ], dtype=np.float32)

    # Batch 5
    image_points = np.array([
        # Example placeholders - REPLACE WITH YOUR ACTUAL PIXEL COORDINATES:
        # [x_pixel_for_P1, y_pixel_for_P1],  # P1: (0,0,0)
        [46,1065],  # P2: (0,0,-1)
        [ 478,792],  # P3: (0,0,1)
        # [x_pixel_for_P4, y_pixel_for_P4],  # P4: (3.3,0,-1)
        [1766, 1021],  # P6: (3,1.58,-0.4)
        [1519,325],  # P5: (3.3,0,1)
        [1380,372],  # P7: (3,1.58,0.4)
        [636,393],
        [318,907]
    ], dtype=np.float32)

    # Batch 6
    image_points = np.array([
        # Example placeholders - REPLACE WITH YOUR ACTUAL PIXEL COORDINATES:
        # [x_pixel_for_P1, y_pixel_for_P1],  # P1: (0,0,0)
        [43,1063],  # P2: (0,0,-1)
        [477,793],  # P3: (0,0,1)
        # [x_pixel_for_P4, y_pixel_for_P4],  # P4: (3.3,0,-1)
        [1764, 1024],  # P6: (3,1.58,-0.4)
        [1519,326],  # P5: (3.3,0,1)
        [1380,374],  # P7: (3,1.58,0.4)
        [528,401],
        [351,878]
    ], dtype=np.float32)

    # Batch 7
    image_points = np.array([
        # Example placeholders - REPLACE WITH YOUR ACTUAL PIXEL COORDINATES:
        # [x_pixel_for_P1, y_pixel_for_P1],  # P1: (0,0,0)
        [ 47,1061],  # P2: (0,0,-1)
        [478,790],  # P3: (0,0,1)
        # [x_pixel_for_P4, y_pixel_for_P4],  # P4: (3.3,0,-1)
        [1766, 1018],  # P6: (3,1.58,-0.4)
        [1523,320],  # P5: (3.3,0,1)
        [1383,368],  # P7: (3,1.58,0.4)
        [510,354],
        [358,876]
    ], dtype=np.float32)

    # Batch 8
    image_points = np.array([
        # Example placeholders - REPLACE WITH YOUR ACTUAL PIXEL COORDINATES:
        # [x_pixel_for_P1, y_pixel_for_P1],  # P1: (0,0,0)
        [48,1055],  # P2: (0,0,-1)
        [478,787],  # P3: (0,0,1)
        # [x_pixel_for_P4, y_pixel_for_P4],  # P4: (3.3,0,-1)
        [1763, 1013],  # P6: (3,1.58,-0.4)
        [1519,318],  # P5: (3.3,0,1)
        [1381,364],  # P7: (3,1.58,0.4)
        [582,366],
        [310,903]
    ], dtype=np.float32)

    # Batch 9
    image_points = np.array([
        # Example placeholders - REPLACE WITH YOUR ACTUAL PIXEL COORDINATES:
        # [x_pixel_for_P1, y_pixel_for_P1],  # P1: (0,0,0)
        [47,1052],  # P2: (0,0,-1)
        [477,783],  # P3: (0,0,1)
        # [x_pixel_for_P4, y_pixel_for_P4],  # P4: (3.3,0,-1)
        [1761, 1011],  # P6: (3,1.58,-0.4)
        [1517, 315],  # P5: (3.3,0,1)
        [1380,361],  # P7: (3,1.58,0.4)
        [492,346],
        [307,903]
    ], dtype=np.float32)

    # Batch 10
    image_points = np.array([
        # Example placeholders - REPLACE WITH YOUR ACTUAL PIXEL COORDINATES:
        # [x_pixel_for_P1, y_pixel_for_P1],  # P1: (0,0,0)
        [40,1052],  # P2: (0,0,-1)
        [ 473,784],  # P3: (0,0,1)
        # [x_pixel_for_P4, y_pixel_for_P4],  # P4: (3.3,0,-1)
        [1755, 1015],  # P6: (3,1.58,-0.4)
        [1515,320],  # P5: (3.3,0,1)
        [1378,366],  # P7: (3,1.58,0.4)
        [516,318],
        [313,895]
    ], dtype=np.float32)

    camera_pos_wcs_guess = np.array(
        [0.5, 0.6, -1])  # X is distance from origin to camera, Z should be small if camera is on side (this is okay-ish azimuth of like 10?)

    #Batch 2
    camera_pos_wcs_guess = np.array(
        [1.6, 0.95,-2])  # X is distance from origin to camera, Z should be small if camera is on side (this is okay-ish azimuth of like 10?)

    # A point the camera is roughly looking at (e.g., center of the shooting area, slightly up)
    look_at_wcs_guess = np.array([1.65,1,1])  # Target is mid-lane (Z=1), mid-height

    # Calculate the rotation matrix for the guess
    # Camera's Z-axis points from camera_pos_wcs_guess towards look_at_wcs_guess
    cam_z_axis = look_at_wcs_guess - camera_pos_wcs_guess
    cam_z_axis /= np.linalg.norm(cam_z_axis)

    # Camera's X-axis is to its right (world Z x world Y, then project)
    world_up = np.array([0, 1, 0])  # WCS Y-axis is up
    cam_x_axis = np.cross(world_up, cam_z_axis)  # This gives a vector to the camera's right
    cam_x_axis /= np.linalg.norm(cam_x_axis)

    # Camera's Y-axis is downwards (relative to its view)
    cam_y_axis = np.cross(cam_z_axis, cam_x_axis)

    # This rotation matrix transforms WCS points to CCS
    R_wcs_to_ccs_guess = np.vstack([cam_x_axis, cam_y_axis, cam_z_axis])

    #initial_rvec_guess, _ = cv2.Rodrigues(R_wcs_to_ccs_guess)
    initial_rvec_guess = np.array([0.26, 0.02, -0.035])

    # The tvec guess is -R_guess * camera_pos_wcs_guess
    initial_tvec_guess = -R_wcs_to_ccs_guess @ camera_pos_wcs_guess.reshape(3, 1)
    initial_tvec_guess = initial_tvec_guess.flatten()

    print(f"Initial rvec guess: {initial_rvec_guess.flatten()}")
    print(f"Initial tvec guess: {initial_tvec_guess.flatten()}")


    #Session 3 Experiment Setup

    if DETECT_ON_FULLY_UNDISTORTED_FRAME:
        # If points are from an image already undistorted with P=K,
        # then K is the correct matrix and distortion is effectively zero.
        dist_coeffs_for_solvepnp = None  # or np.zeros((4,1))
    else:
        # If points are from the original distorted image
        dist_coeffs_for_solvepnp = dist_coeffs_val

    try:
        success, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix,
                                           dist_coeffs_for_solvepnp,
                                           flags=cv2.SOLVEPNP_SQPNP)  # cv2.SOLVEPNP_SQPNP might be faster if available

        success, rvec, tvec = cv2.solvePnP(
            world_points, image_points, camera_matrix,
            dist_coeffs_for_solvepnp,
            #rvec=initial_rvec_guess,  # Pass the guess for rotation
            tvec=initial_tvec_guess,  # Pass the guess for translation
            useExtrinsicGuess=True,  # CRITICAL: Use the guess
            flags=cv2.SOLVEPNP_EPNP  # Good for refining a guess
        )
        #print(f"rvec: {rvec}")
        #rvec= np.array([[3.0943],[-0.05],[-0.20304]]) #-0.07 was okayish
        # rvec, tvec = cv2.solvePnPRefineLM(
        #     world_points, image_points, camera_matrix,
        #     dist_coeffs_for_solvepnp,
        #     rvec=initial_rvec_guess,  # Pass the guess for rotation
        #     tvec=initial_tvec_guess,  # Pass the guess for translation
        # )
        # success = True

        if success:
            R_wcs_to_ccs, _ = cv2.Rodrigues(rvec)
            # rvec_y_correction = np.array([0, -0.1, 0])
            # R_y,_ = cv2.Rodrigues(rvec_y_correction)
            # R_new = R_y @ R_wcs_to_ccs
            # R_wcs_to_ccs,_ = cv2.Rodrigues(R_new)

            #tvec_wcs_to_ccs = np.array([[tvec[0]],[tvec[1]],[tvec[2]]])
            tvec_wcs_to_ccs = np.array([[-1.6], [0.97], [3.9]]) #shifted z????
            wcs_setup_done = True

            test_world_point = np.array([0.65,1.12,0.93], dtype=np.float32)  # Your ball release for Batch 1
            projected_image_point, _ = cv2.projectPoints(test_world_point.reshape(1, 1, 3), rvec, tvec, camera_matrix,
                                                         dist_coeffs_for_solvepnp)
            print(
                f"\nSanity Check: World point {test_world_point} projects to image point: {projected_image_point.flatten().round(1)}")

            print("WCS setup successful. R_wcs_to_ccs and tvec_wcs_to_ccs computed.")
            print("R_wcs_to_ccs:\n", R_wcs_to_ccs)
            print("tvec_wcs_to_ccs:\n", tvec_wcs_to_ccs)
        else:
            print("WCS setup failed (solvePnP returned False).")
            wcs_setup_done = False
    except Exception as e:
        print(f"Error during solvePnP: {e}")
        wcs_setup_done = False
    return wcs_setup_done


def ccs_to_wcs(Pc_ccs):
    if not wcs_setup_done or R_wcs_to_ccs is None or tvec_wcs_to_ccs is None:
        return None
    # Pc_wcs = R_ccs_to_wcs * (Pc_ccs - tvec_wcs_to_ccs_origin_in_ccs)
    # R_ccs_to_wcs = R_wcs_to_ccs.T
    # tvec_ccs_to_wcs_origin_in_wcs = -R_wcs_to_ccs.T @ tvec_wcs_to_ccs
    # Pc_wcs = R_wcs_to_ccs.T @ Pc_ccs + (-R_wcs_to_ccs.T @ tvec_wcs_to_ccs)
    # Pc_wcs = R_wcs_to_ccs.T @ (Pc_ccs - tvec_wcs_to_ccs)

    # Ensure Pc_ccs is a column vector (3,1)
    Pc_ccs_col = Pc_ccs.reshape((3, 1))
    Pc_wcs_col = R_wcs_to_ccs.T @ (Pc_ccs_col - tvec_wcs_to_ccs)
    return Pc_wcs_col.flatten()  # Return as a flat array (1D)

def setup_kalman_filter(dt=1.0 / 30, noise_acceleration_variance = 5000):
    kf = cv2.KalmanFilter(4, 2)  # 4 state vars (x,y,vx,vy), 2 measurement vars (x,y)
    kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)  # Measures x,y

    # Simplified process noise covariance, q is a general factor
    # Adjust q_pos_factor and q_vel_factor based on expected movement variance
    q_pos_factor = 1.0  # Factor for position variance in process noise
    q_vel_factor = 1.0  # Factor for velocity variance in process noise

    kf.processNoiseCov = np.array([
        [dt ** 4 / 4 * q_pos_factor, 0, dt ** 3 / 2 * q_pos_factor, 0],
        [0, dt ** 4 / 4 * q_pos_factor, 0, dt ** 3 / 2 * q_pos_factor],
        [dt ** 3 / 2 * q_pos_factor, 0, dt ** 2 * q_vel_factor, 0],
        [0, dt ** 3 / 2 * q_pos_factor, 0, dt ** 2 * q_vel_factor]
    ], dtype=np.float32) * noise_acceleration_variance

    #Uncertainty of YOLO detections (change based on noisiness of pixel detections)
    kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], dtype=np.float32) * KF_MEASUREMENT_NOISE_SCALE
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 500  # High initial uncertainty
    return kf


def undistort_point(point_xy, K_matrix, D_coeffs, is_fisheye=True):
    if point_xy is None: return None
    dist_np = np.array([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
    undist_np = cv2.fisheye.undistortPoints(dist_np, K_matrix, D_coeffs, P=K_matrix) if is_fisheye \
        else cv2.undistortPoints(dist_np, K_matrix, D_coeffs, P=K_matrix)
    if undist_np is not None: return (int(round(undist_np[0][0][0])), int(round(undist_np[0][0][1])))
    return None


def preprocess_frame_ccs(frame_orig, K_matrix, D_coeffs_val, map1_val, map2_val, do_full_undistort_val):
    if do_full_undistort_val and map1_val is not None and map2_val is not None:  # map2_val check added
        return cv2.remap(frame_orig, map1_val, map2_val, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return frame_orig


def get_processed_detections_ccs(current_frame_num_val, frame_to_detect_on_val, ball_model_obj, pose_model_obj,
                                 id_ball_class_val, current_ball_conf_val, K_matrix_val, D_coeffs_val,
                                 do_undistort_points_val, is_full_frame_undistorted_already,
                                 current_kf_trajectory_pixels_val,  # List of dicts {'frame', 'px_coord', 'type'}
                                 max_pixel_disp_val, is_release_detected_currently):
    ball_results = ball_model_obj(frame_to_detect_on_val, conf=current_ball_conf_val, verbose=False,
                                  classes=[id_ball_class_val])
    ball_bbox_detected = utils.get_ball_bbox(ball_results, id_ball_class_val)
    ball_center_undistorted_px_final = None
    ball_bbox_for_logic_val = ball_bbox_detected
    used_low_conf_heuristic_val_flag = False

    if ball_bbox_detected is not None:
        center_x_det = (ball_bbox_detected[0] + ball_bbox_detected[2]) / 2.0
        center_y_det = (ball_bbox_detected[1] + ball_bbox_detected[3]) / 2.0

        temp_ball_center_processed_px = None
        if do_undistort_points_val and not is_full_frame_undistorted_already:
            temp_ball_center_processed_px = undistort_point((center_x_det, center_y_det), K_matrix_val, D_coeffs_val)
        else:
            temp_ball_center_processed_px = (int(round(center_x_det)), int(round(center_y_det)))

        if temp_ball_center_processed_px is not None:
            if is_release_detected_currently and len(current_kf_trajectory_pixels_val) > 0:
                # Use the last known KF point for proximity check
                last_known_kf_px = current_kf_trajectory_pixels_val[-1]['px_coord']
                dist_sq = (temp_ball_center_processed_px[0] - last_known_kf_px[0]) ** 2 + \
                          (temp_ball_center_processed_px[1] - last_known_kf_px[1]) ** 2
                if dist_sq <= max_pixel_disp_val ** 2:
                    ball_center_undistorted_px_final = temp_ball_center_processed_px
                    if current_ball_conf_val == POST_RELEASE_BALL_CONF: used_low_conf_heuristic_val_flag = True
                else:
                    ball_bbox_for_logic_val = None  # Rejected raw detection by proximity to KF track
            else:  # Pre-release or first point post-release (no KF trajectory yet to check against)
                ball_center_undistorted_px_final = temp_ball_center_processed_px
        else:  # Undistortion failed
            ball_bbox_for_logic_val = None
    # else: No ball detected

    pose_results_val = pose_model_obj(frame_to_detect_on_val, conf=0.35, verbose=False)
    shooter_wrist_undistorted_px_final = None
    if ball_bbox_for_logic_val is not None:  # Only proceed if we have a valid ball bbox for context

        # Original
        # wrist_detected_val, _ = utils.get_likely_shooter_wrist_and_person_idx(pose_results_val, ball_bbox_for_logic_val,
        #                                                                       SHOOTING_WRIST_KP_INDEX,
        #                                                                       MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION)
        print(f"Frame Wrist Detection: {current_frame_num_val}")
        wrist_detected_val, person_idx_val= utils.get_likely_shooter_wrist_and_person_idx(
            pose_results_list=pose_results_val,  # This is the list of YOLO Results objects
            ball_bbox=ball_bbox_for_logic_val,
            shooting_arm_wrist_idx=SHOOTING_ARM_WRIST_IDX,
            shooting_arm_elbow_idx=SHOOTING_ARM_ELBOW_IDX,
            shooting_arm_shoulder_idx=SHOOTING_ARM_SHOULDER_IDX,
            max_dist_threshold=MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION,  # Your existing global
        )

        if wrist_detected_val is not None:
            if do_undistort_points_val and not is_full_frame_undistorted_already:
                shooter_wrist_undistorted_px_final = undistort_point(wrist_detected_val, K_matrix_val, D_coeffs_val)
            else:
                shooter_wrist_undistorted_px_final = (int(wrist_detected_val[0]), int(wrist_detected_val[1]))

    return ball_center_undistorted_px_final, ball_bbox_for_logic_val, shooter_wrist_undistorted_px_final, used_low_conf_heuristic_val_flag


def find_release_candidate(frame_num_val, ball_bbox_l_val, wrist_pos_undist_px, release_buffer_list_val):
    is_shot_stance = False
    if ball_bbox_l_val is not None and wrist_pos_undist_px is not None:
        ball_center_y = (ball_bbox_l_val[1] + ball_bbox_l_val[3]) / 2
        # Use frame_height from main scope
        if ball_center_y < frame_height_g * MIN_BALL_HEIGHT_FOR_SHOT_GATHER and \
                wrist_pos_undist_px[1] < frame_height_g * MIN_WRIST_HEIGHT_FOR_SHOT_GATHER:
            is_shot_stance = True

    candidate_frame = -1  # Return -1 if no candidate found

    if is_shot_stance:
        wrist_in_ball = utils.is_point_inside_bbox(wrist_pos_undist_px, ball_bbox_l_val,
                                                   margin=utils.WRIST_BALL_PROXIMITY_MARGIN)
        if not wrist_in_ball:
            release_buffer_list_val.append(frame_num_val)
            if len(release_buffer_list_val) > utils.CONFIRMATION_FRAMES_FOR_RELEASE:
                release_buffer_list_val.pop(0)
            if len(release_buffer_list_val) == utils.CONFIRMATION_FRAMES_FOR_RELEASE:
                candidate_frame = release_buffer_list_val[0]
                release_buffer_list_val.clear()  # Reset after finding one
        else:
            release_buffer_list_val.clear()
    else:
        release_buffer_list_val.clear()

    return candidate_frame, release_buffer_list_val


def update_trajectory_and_estimate_ccs(current_release_state_val,
                                       ball_point_from_kf_px_dict,
                                       current_Zc_at_release, video_fps_val):
    # Globals MODIFIED or used
    global estimated_speed_mps, estimated_angle_deg_XY_ccs, estimated_angle_deg_elev_ccs, shot_count # CCS estimates
    global trajectory_points_kf_pixels_undistorted, trajectory_points_ccs
    # --- WCS globals ---
    global trajectory_points_wcs, estimated_speed_mps_wcs, estimated_angle_deg_elev_wcs, estimated_angle_deg_azim_wcs

    if current_release_state_val and ball_point_from_kf_px_dict is not None and current_Zc_at_release is not None:
        frame_num_val = ball_point_from_kf_px_dict['frame']
        u_kf, v_kf = ball_point_from_kf_px_dict['px_coord']
        point_type = ball_point_from_kf_px_dict['type']

        # ... (KF pixel trajectory update logic remains the same) ...
        if not trajectory_points_kf_pixels_undistorted or \
                trajectory_points_kf_pixels_undistorted[-1]['frame'] != frame_num_val:
            trajectory_points_kf_pixels_undistorted.append(ball_point_from_kf_px_dict)


        # Convert KF point to CCS and append
        current_Pc_ccs_kf = None
        if not trajectory_points_ccs or trajectory_points_ccs[-1]['frame'] != frame_num_val:
            xn_kf = (u_kf - cx) / fx; yn_kf = (v_kf - cy) / fy
            current_Pc_ccs_kf = np.array([xn_kf * current_Zc_at_release, yn_kf * current_Zc_at_release, current_Zc_at_release])
            trajectory_points_ccs.append({'frame': frame_num_val, 'Pc_ccs': current_Pc_ccs_kf, 'px_coord_type': point_type})

            # --- Transform to WCS and append ---
            if wcs_setup_done and current_Pc_ccs_kf is not None: # Check current_Pc_ccs_kf
                Pc_wcs_kf = ccs_to_wcs(current_Pc_ccs_kf)
                if Pc_wcs_kf is not None:
                    trajectory_points_wcs.append({'frame': frame_num_val, 'Pc_wcs': Pc_wcs_kf, 'px_coord_type': point_type})

        # Naive parameter estimation (CCS - original)
        if estimated_speed_mps == 0.0 and len(trajectory_points_ccs) >= 1 + FRAMES_FOR_NAIVE_ESTIMATION:
            p_start_ccs_info = trajectory_points_ccs[0]
            p_end_ccs_info = trajectory_points_ccs[FRAMES_FOR_NAIVE_ESTIMATION] # Ensure this index is valid
            Pc_s_ccs = p_start_ccs_info['Pc_ccs']; Pc_e_ccs = p_end_ccs_info['Pc_ccs']
            delta_Pc_ccs_val = Pc_e_ccs - Pc_s_ccs
            d_frames_ccs = p_end_ccs_info['frame'] - p_start_ccs_info['frame']

            if video_fps_val > 0 and d_frames_ccs > 0:
                d_t_s_ccs = d_frames_ccs / video_fps_val
                Vc_avg_ccs_val = delta_Pc_ccs_val / d_t_s_ccs
                estimated_speed_mps = np.linalg.norm(Vc_avg_ccs_val)
                # CCS: X right, Y down, Z forward
                mag_XZ_ccs = math.sqrt(Vc_avg_ccs_val[0] ** 2 + Vc_avg_ccs_val[2] ** 2)
                if mag_XZ_ccs < 1e-6: mag_XZ_ccs = 1e-6
                estimated_angle_deg_elev_ccs = math.degrees(math.atan2(-Vc_avg_ccs_val[1], mag_XZ_ccs)) # -Yc because Yc is down
                estimated_angle_deg_XY_ccs = math.degrees(math.atan2(Vc_avg_ccs_val[0], Vc_avg_ccs_val[2])) # Azimuth in XZ_ccs plane from Z_ccs

        # --- Naive parameter estimation (WCS) ---
        # WCS: X towards hoop, Y up, Z right
        if wcs_setup_done and estimated_speed_mps_wcs == 0.0 and len(trajectory_points_wcs) >= 1 + FRAMES_FOR_NAIVE_ESTIMATION:
            p_start_wcs_info = trajectory_points_wcs[0]
            p_end_wcs_info = trajectory_points_wcs[FRAMES_FOR_NAIVE_ESTIMATION] # Ensure this index is valid

            Pc_s_wcs = p_start_wcs_info['Pc_wcs']; Pc_e_wcs = p_end_wcs_info['Pc_wcs']
            delta_Pc_wcs_val = Pc_e_wcs - Pc_s_wcs
            d_frames_wcs = p_end_wcs_info['frame'] - p_start_wcs_info['frame']

            if video_fps_val > 0 and d_frames_wcs > 0:
                d_t_s_wcs = d_frames_wcs / video_fps_val
                Vc_avg_wcs_val = delta_Pc_wcs_val / d_t_s_wcs # [VX_wcs, VY_wcs, VZ_wcs]
                estimated_speed_mps_wcs = np.linalg.norm(Vc_avg_wcs_val)

                # Elevation angle (angle with XZ_wcs plane, i.e., the ground plane)
                # VY_wcs is the vertical component, sqrt(VX_wcs^2 + VZ_wcs^2) is projection on XZ_wcs plane.
                mag_XZ_wcs = math.sqrt(Vc_avg_wcs_val[0]**2 + Vc_avg_wcs_val[2]**2)
                if mag_XZ_wcs < 1e-6: mag_XZ_wcs = 1e-6 # Avoid division by zero if shot is purely vertical
                estimated_angle_deg_elev_wcs = math.degrees(math.atan2(Vc_avg_wcs_val[1], mag_XZ_wcs))

                # Azimuth angle (in XZ_wcs plane, angle from X_wcs-axis towards Z_wcs-axis)
                # This represents side-to-side deviation. X_wcs is straight towards hoop.
                # atan2(VZ_wcs, VX_wcs)
                estimated_angle_deg_azim_wcs = math.degrees(math.atan2(Vc_avg_wcs_val[2], Vc_avg_wcs_val[0]))

                print(
                    f"  NAIVE WCS (Shot {shot_count}, frames {p_start_wcs_info['frame']}-{p_end_wcs_info['frame']}): "
                    f"Speed={estimated_speed_mps_wcs:.2f}m/s, Elev(Y/XZ)={estimated_angle_deg_elev_wcs:.1f}deg, Azim(Z/X)={estimated_angle_deg_azim_wcs:.1f}deg")

def draw_annotations_on_frame_ccs(frame_to_annotate_val, current_frame_number_for_drawing,
                                  ball_bbox_d_val, raw_ball_center_undist_px_val,
                                  wrist_pos_undist_px_val, is_release_detected_val,
                                  release_info_dict_val, traj_kf_pix_list_of_dicts,
                                  traj_raw_pix_list_of_dicts, shot_num_val,
                                  est_speed_val_ccs, est_angle_elev_val_ccs, # CCS estimates
                                  # WCS estimates are accessed via global variables in this example
                                  used_low_conf_flag_val,
                                  kf_current_prediction_px_val=None):

    # Draw raw detected bbox (from current frame's detection)
    if ball_bbox_d_val is not None:
        l, t, r, b = [int(c) for c in ball_bbox_d_val]
        color_bbox = (0, 165, 255) if used_low_conf_flag_val else (
        255, 100, 100)  # Orange if low conf for this raw detection
        cv2.rectangle(frame_to_annotate_val, (l, t), (r, b), color_bbox, 1)

    # Draw raw detected (undistorted) center for the CURRENT frame
    if raw_ball_center_undist_px_val is not None:
        cv2.circle(frame_to_annotate_val, raw_ball_center_undist_px_val, 4, (255, 0, 255),
                   -1)  # Magenta for raw detection point

    # Draw current KF point (corrected or predicted if no good raw detection was used for correction)
    current_kf_point_to_draw = None
    current_kf_point_type = 'unknown'
    if traj_kf_pix_list_of_dicts:  # Check if list is not empty
        # Find the KF point for the current frame, if it exists in the trajectory list
        kf_point_for_current_frame = next(
            (item for item in reversed(traj_kf_pix_list_of_dicts) if item['frame'] == current_frame_number_for_drawing),
            None)
        if kf_point_for_current_frame:
            current_kf_point_to_draw = kf_point_for_current_frame['px_coord']
            current_kf_point_type = kf_point_for_current_frame['type']
        elif kf_current_prediction_px_val and is_release_detected_val:  # Fallback to raw KF prediction if no KF point was added to traj list for this frame
            current_kf_point_to_draw = kf_current_prediction_px_val
            current_kf_point_type = 'kf_pred_only_not_in_traj'

    if current_kf_point_to_draw:
        color_kf = (0, 255, 0)  # Green for 'kf_corr' or 'kf_init'
        if 'pred' in current_kf_point_type:
            color_kf = (255, 255, 0)  # Cyan for 'kf_pred_gated', 'kf_pred_miss'
        elif current_kf_point_type == 'kf_pred_only_not_in_traj':
            color_kf = (255, 165, 0)  # Orange for pure prediction not added to trajectory

        cv2.circle(frame_to_annotate_val, current_kf_point_to_draw, 6, color_kf, -1)
        # cv2.putText(frame_to_annotate_val, current_kf_point_type, (current_kf_point_to_draw[0]+10, current_kf_point_to_draw[1]), cv2.FONT_HERSHEY_PLAIN, 0.8, color_kf, 1)

    if wrist_pos_undist_px_val is not None: cv2.circle(frame_to_annotate_val, tuple(wrist_pos_undist_px_val), 7,
                                                       (0, 255, 255), -1)  # Yellow for wrist

    # Draw RAW trajectory dots (magenta dots)
    if is_release_detected_val and len(traj_raw_pix_list_of_dicts) >= 1:
        for item_info in traj_raw_pix_list_of_dicts:
            # Draw older points smaller/fainter if desired
            cv2.circle(frame_to_annotate_val, item_info['px_coord'], 2, (200, 0, 200), -1)  # Darker Magenta dots

    # Draw KF trajectory line (main green line)
    if is_release_detected_val and len(traj_kf_pix_list_of_dicts) >= 2:
        pixel_points_for_kf_line = [item['px_coord'] for item in traj_kf_pix_list_of_dicts]
        pts_np_kf = np.array(pixel_points_for_kf_line, dtype=np.int32)
        cv2.polylines(frame_to_annotate_val, [pts_np_kf], False, (0, 255, 0), 2)  # Green line

    # Text annotations
    if is_release_detected_val:
        if release_info_dict_val: cv2.putText(frame_to_annotate_val,
                                              f"S{shot_num_val} Rls:Fr{release_info_dict_val['frame_no']}",
                                              (10, 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if traj_kf_pix_list_of_dicts and traj_kf_pix_list_of_dicts[0]['px_coord'] is not None:
            cv2.circle(frame_to_annotate_val, traj_kf_pix_list_of_dicts[0]['px_coord'], 15, (0, 0, 255),
                       2)  # Release marker

        if est_speed_val_ccs > 0:  # CCS estimates
            cv2.putText(frame_to_annotate_val, f"Speed(CCS): {est_speed_val_ccs:.2f}m/s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            cv2.putText(frame_to_annotate_val, f"Elev(CCS): {est_angle_elev_val_ccs:.1f}deg", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        start_y_offset = 120  # Start WCS text below CCS text
        if wcs_setup_done and estimated_speed_mps_wcs > 0:  # Naive WCS
            cv2.putText(frame_to_annotate_val, f"Speed(WCS-N): {estimated_speed_mps_wcs:.2f}m/s", (10, start_y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 100), 2)
            cv2.putText(frame_to_annotate_val, f"Elev(WCS-N): {estimated_angle_deg_elev_wcs:.1f}deg",
                        (10, start_y_offset + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 100), 2)
            cv2.putText(frame_to_annotate_val, f"Azim(WCS-N): {estimated_angle_deg_azim_wcs:.1f}deg",
                        (10, start_y_offset + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 100), 2)
            start_y_offset += 90

        # --- Display WCS estimates (using global variables for simplicity here) ---
        global fitted_params_dict  # Use the dictionary
        if fitted_params_dict and fitted_params_dict.get("success") and \
                fitting_performed_for_shot == shot_num_val:  # Check if fit was for current shot

            start_y_offset_fit = 120  # Continue from where naive WCS text left off
            # (Assuming start_y_offset was updated after naive WCS text)

            cv2.putText(frame_to_annotate_val, f"Speed(WCS-Fit): {fitted_params_dict['speed_mps_wcs']:.2f}m/s",
                        (10, start_y_offset_fit),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame_to_annotate_val, f"Elev(WCS-Fit): {fitted_params_dict['angle_deg_elev_wcs']:.1f}deg",
                        (10, start_y_offset_fit + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame_to_annotate_val, f"Azim(WCS-Fit): {fitted_params_dict['angle_deg_azim_wcs']:.1f}deg",
                        (10, start_y_offset_fit + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        # ... (existing pre-release text) ...
        cv2.putText(frame_to_annotate_val, "Awaiting Release", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        if not wcs_setup_done:
            cv2.putText(frame_to_annotate_val, "WCS NOT READY", (frame_to_annotate_val.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame_to_annotate_val


def fitting_wcs_tracking(all_shots_data_list, video_fps, session_dir):
    """
    Loops through all completed shots, performs fitting, plotting, and saves results
    into a structured directory for each shot.
    """
    print(f"\n--- Starting Final Analysis for {len(all_shots_data_list)} Completed Shots ---")
    if not all_shots_data_list:
        print("No completed shots were recorded for analysis.")
        return

    #video_basename = os.path.splitext(os.path.basename(video_path))[0]

    for shot_data in all_shots_data_list:
        shot_id = shot_data["shot_id"]

        # --- Define the specific output directory for this shot ---
        shot_output_dir = os.path.join(session_dir, f"shot_{shot_id:03d}")
        os.makedirs(shot_output_dir, exist_ok=True)  # Ensure it exists, though it should

        print(f"\n--- Analyzing Shot {shot_id} (Output to: {shot_output_dir}) ---")

        if wcs_setup_done and len(shot_data["trajectory_wcs"]) >= FRAMES_FOR_NAIVE_ESTIMATION + 1:
            # Prepare initial guesses for the optimizer
            initial_p0_guess_fit = shot_data["trajectory_wcs"][0]['Pc_wcs']

            initial_v_speed_angle_guess_fit = [
                shot_data["naive_speed_wcs"] if shot_data["naive_speed_wcs"] > 0 else 8.0,
                shot_data["naive_elev_wcs"] if shot_data["naive_elev_wcs"] != 0 else 45.0,
                shot_data["naive_azim_wcs"] if shot_data["naive_azim_wcs"] != 0 else 0.0
            ]

            # Perform the fitting
            fitted_params_result = tu.fit_shot_trajectory_parameters(
                shot_data["trajectory_wcs"],
                video_fps,
                initial_p0_guess_fit,
                initial_v_speed_angle_guess_fit
            )

            # Store the results back into the shot's data dictionary
            shot_data["fitted_params"] = fitted_params_result

            # --- Save all data for this shot inside its dedicated folder ---

            # 1. Save KF WCS trajectory
            wcs_kf_traj_output_path = os.path.join(shot_output_dir, "mono_wcs_trajectory.txt")
            with open(wcs_kf_traj_output_path, 'w') as f_out:
                f_out.write("frame,x_wcs,y_wcs,z_wcs,type\n")
                for pt_info in shot_data["trajectory_wcs"]:
                    f_out.write(
                        f"{pt_info['frame']},{pt_info['Pc_wcs'][0]},{pt_info['Pc_wcs'][1]},{pt_info['Pc_wcs'][2]},{pt_info['px_coord_type']}\n")
            print(f"  - Saved KF WCS trajectory for shot {shot_id}.")

            # 2. Save fitted parameters and plot if fitting was successful
            if fitted_params_result and fitted_params_result.get("success"):
                # Save fitted parameters text file
                fitted_params_output_path = os.path.join(shot_output_dir, "mono_fitted_params.txt")
                with open(fitted_params_output_path, 'w') as f_out:
                    params = shot_data["fitted_params"]
                    initial_p0_guess = shot_data["trajectory_wcs"][0]['Pc_wcs']
                    initial_v_guess = [shot_data["naive_speed_wcs"], shot_data["naive_elev_wcs"],
                                       shot_data["naive_azim_wcs"]]
                    f_out.write(f"Fitted_P0_wcs: {params['P0_wcs'].tolist()}\n")
                    f_out.write(f"Fitted_V0_wcs_components: {params['V0_wcs_components'].tolist()}\n")
                    f_out.write(f"Fitted_Speed_mps_wcs: {params['speed_mps_wcs']}\n")
                    f_out.write(f"Fitted_Angle_Elev_wcs: {params['angle_deg_elev_wcs']}\n")
                    f_out.write(f"Fitted_Angle_Azim_wcs: {params['angle_deg_azim_wcs']}\n")
                    f_out.write(f"KB_BALL_used: {tu.KB_BALL}\n")
                    f_out.write(f"Initial_P0_Guess: {initial_p0_guess.tolist()}\n")
                    f_out.write(f"Initial_V_Angle_Guess: {initial_v_guess}\n")
                print(f"  - Saved fitted parameters for shot {shot_id}.")

                # Generate and save the plot
                plot_court_dimensions = {
                    "ft_to_rim_x": WCS_FT_LINE_TO_RIM_CENTER_X, "rim_y": WCS_RIM_HEIGHT_Y,
                    "rim_diameter": WCS_RIM_RADIUS * 2, "backboard_x": WCS_BACKBOARD_X,
                    "backboard_height": 1.0668, "bb_inner_rect_height": WCS_BB_INNER_RECT_HEIGHT
                }
                tu.plot_fitted_trajectory_wcs(
                    observed_wcs_positions=shot_data["trajectory_wcs"],
                    fitted_P0=fitted_params_result["P0_wcs"],
                    fitted_V0_components=fitted_params_result["V0_wcs_components"],
                    shot_number=shot_id,
                    video_fps=video_fps,
                    # Pass the specific shot directory for the plot
                    output_dir=shot_output_dir,
                    # The plot function can use a generic name now
                    video_basename="fitted_trajectory_plot",
                    court_dims=plot_court_dimensions
                )
                print(f"  - Saved trajectory plot for shot {shot_id}.")
            else:
                print(f"  - Fitting failed for shot {shot_id}. No parameters or plot saved.")
        else:
            print(f"  - Not enough data or WCS not ready for shot {shot_id}. Skipping fitting and saving.")


# --- Main Processing Loop ---
if __name__ == "__main__":
    os.makedirs(RELEASE_FRAME_OUTPUT_DIR, exist_ok=True)

    MODEL_WEIGHTS_DIR = os.path.join(BASE_DIR, "Yolo-Weights")
    BALL_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "model2.pt")
    POSE_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR,'yolov8n-pose.pt')

    VIDEO_INPUT_DIR = os.path.join(BASE_DIR, "data", "footage")  # Define VIDEO_INPUT_DIR globally
    # VIDEO_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "pipelined_wcs", "output_video_wcs_fit")  # Changed output dir slightly
    # os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

    DATA_OUTPUT_ROOT = os.path.join(BASE_DIR, "data", "experiment_data_all_batches")
    os.makedirs(DATA_OUTPUT_ROOT, exist_ok=True)

    video_path = os.path.join(VIDEO_INPUT_DIR, "batch1.mp4")  # Example video

    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # A specific directory for all outputs from this video file
    session_output_dir = os.path.join(DATA_OUTPUT_ROOT, video_basename)
    os.makedirs(session_output_dir, exist_ok=True)



    ball_model_g = YOLO(BALL_MODEL_PATH)
    pose_model_g = YOLO(POSE_MODEL_PATH)
    load_release_model(MODEL_PATH)

    cap_g = cv2.VideoCapture(video_path)
    if not cap_g.isOpened(): cap_g = cv2.VideoCapture(video_path, cv2.CAP_MSMF)  # Fallback
    if not cap_g.isOpened(): exit(f"Error opening video: {video_path}")

    frame_width_g = int(cap_g.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_g = int(cap_g.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps_g = cap_g.get(cv2.CAP_PROP_FPS)
    if not (0 < video_fps_g < 1000): video_fps_g = 30.0  # More robust FPS check, limit to 200

    id_of_ball_g = None
    for cid, name in ball_model_g.names.items():
        if name.lower() == TARGET_BALL_CLASS_NAME: id_of_ball_g = cid; break
    if id_of_ball_g is None: exit(f"Class '{TARGET_BALL_CLASS_NAME}' ID not found in model.")

    map1_g, map2_g = None, None
    if DETECT_ON_FULLY_UNDISTORTED_FRAME:
        map1_g, map2_g = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (frame_width_g, frame_height_g),
                                                             cv2.CV_16SC2)

    # --- Setup WCS ---
    # Call setup_wcs once, K is global. D might be needed if not using fully undistorted frames for solvePnP points
    if not setup_wcs(K, D):  # Pass D in case you change DETECT_ON_FULLY_UNDISTORTED_FRAME
        print("WARNING: WCS setup failed. WCS transformations will not be available.")
        # Decide if you want to exit or continue without WCS
        # exit("Exiting due to WCS setup failure.")

    # --- Main loop state variables ---
    frame_count = 0
    shot_count = 0
    shot_active_timer = 0
    release_detected_in_shot = False
    release_frame_info_dict = None
    potential_release_buffer = []
    frame_buffer = []  # For storing recent frames for model inference
    buffer_size = INFERENCE_WINDOW_BEFORE + INFERENCE_WINDOW_AFTER + 5
    cooldown_timer = 0

    trajectory_points_raw_pixels_undistorted = []
    trajectory_points_kf_pixels_undistorted = []
    trajectory_points_ccs = []  # List of {'frame', 'Pc_ccs', ...}
    trajectory_points_wcs = []  # List of {'frame', 'Pc_wcs', ...} # NEW
    all_shots_data = []  # NEW: To store data for all completed shots

    shot_video_writer = None  # This will hold the VideoWriter for the current active shot

    Zc_at_release = None
    # CCS estimates
    estimated_speed_mps = 0.0
    estimated_angle_deg_XY_ccs = 0.0  # This was your side angle in CCS XY plane
    estimated_angle_deg_elev_ccs = 0.0
    # WCS estimates
    estimated_speed_mps_wcs = 0.0  # NEW
    estimated_angle_deg_elev_wcs = 0.0  # NEW - Elevation from ground
    estimated_angle_deg_azim_wcs = 0.0  # NEW - Azimuth from Y_wcs (towards basket)

    kf = setup_kalman_filter(dt=1.0 / video_fps_g, noise_acceleration_variance=NOISE_ACCELERATION_VARIANCE)
    kf_initialized = False
    missed_detections_count_kf = 0
    kf_current_predicted_pixel_point_for_drawing = None

    # --- Main Loop ---
    while cap_g.isOpened():
        ret, frame_original_main = cap_g.read()
        if not ret: break
        frame_count += 1



        # 1. Perform detections on the current frame
        current_frame_to_detect = preprocess_frame_ccs(frame_original_main, K, D, map1_g, map2_g,
                                                       DETECT_ON_FULLY_UNDISTORTED_FRAME)
        annotated_frame = current_frame_to_detect.copy()

        active_ball_conf = POST_RELEASE_BALL_CONF if release_detected_in_shot else PRE_RELEASE_BALL_CONF

        ball_center_px, ball_bbox, wrist_pos_px, low_conf_flag = get_processed_detections_ccs(
            frame_count, current_frame_to_detect, ball_model_g, pose_model_g, id_of_ball_g,
            active_ball_conf, K, D, UNDISTORT_POINTS_AFTER_DETECTION, DETECT_ON_FULLY_UNDISTORTED_FRAME,
            trajectory_points_kf_pixels_undistorted, MAX_PIXEL_DISPLACEMENT_POST_RELEASE, release_detected_in_shot
        )

        # 2. Maintain frame buffer for model refinement
        frame_buffer.append({
            "frame_num": frame_count,
            "image": frame_original_main.copy(),
            "ball_bbox": ball_bbox,
            "ball_center": ball_center_px,
            "wrist_pos": wrist_pos_px,
            "low_conf_flag": low_conf_flag
        })
        if len(frame_buffer) > buffer_size: frame_buffer.pop(0)

        # 3. State Machine Logic
        if not release_detected_in_shot:
            # --- State: PRE-RELEASE (Looking for a new shot) ---
            if cooldown_timer > 0:
                cooldown_timer -= 1
            else:
                candidate_frame, potential_release_buffer = find_release_candidate(
                    frame_count, ball_bbox, wrist_pos_px, potential_release_buffer
                )

                if candidate_frame != -1:
                    model_refined_frame = refine_release_frame_with_model(
                        candidate_frame, frame_buffer
                    )

                    #also creates video writer
                    if initiate_shot_tracking(model_refined_frame, frame_buffer, ball_model_g, pose_model_g):
                        cooldown_timer = SHOT_COOLDOWN_PERIOD
                    else:
                        cooldown_timer = 30
        else:
            # --- State: POST-RELEASE (Shot is active) ---
            shot_active_timer -= 1

            ball_point_from_kf_for_ccs_update = None
            if kf_initialized:
                predicted_kf_state = kf.predict()
                kf_current_predicted_pixel_point_for_drawing = (
                int(round(predicted_kf_state[0, 0])), int(round(predicted_kf_state[1, 0])))

                if ball_center_px is not None:
                    dist_sq_to_kf_pred = (ball_center_px[0] - kf_current_predicted_pixel_point_for_drawing[0]) ** 2 + (
                                ball_center_px[1] - kf_current_predicted_pixel_point_for_drawing[1]) ** 2
                    if dist_sq_to_kf_pred <= MAX_PIXEL_DISPLACEMENT_POST_RELEASE ** 2:
                        measurement = np.array([[float(ball_center_px[0])], [float(ball_center_px[1])]],
                                               dtype=np.float32)
                        corrected_kf_state = kf.correct(measurement)
                        ball_point_from_kf_for_ccs_update = {'frame': frame_count, 'px_coord': (
                        int(round(corrected_kf_state[0, 0])), int(round(corrected_kf_state[1, 0]))), 'type': 'kf_corr'}
                        missed_detections_count_kf = 0
                    else:
                        missed_detections_count_kf += 1
                else:
                    missed_detections_count_kf += 1

                if ball_point_from_kf_for_ccs_update is None and missed_detections_count_kf <= MAX_CONSECUTIVE_MISSES_KF:
                    ball_point_from_kf_for_ccs_update = {'frame': frame_count,
                                                         'px_coord': kf_current_predicted_pixel_point_for_drawing,
                                                         'type': 'kf_pred'}

                if ball_point_from_kf_for_ccs_update is not None:
                    update_trajectory_and_estimate_ccs(True, ball_point_from_kf_for_ccs_update, Zc_at_release,
                                                       video_fps_g)

            shot_ended = False
            if shot_active_timer <= 0:
                print(f"Shot {shot_count} ended (duration timer expired).")
                shot_ended = True
            elif missed_detections_count_kf > MAX_CONSECUTIVE_MISSES_KF:
                print(f"Shot {shot_count} ended (prolonged ball loss).")
                shot_ended = True

            if shot_ended:
                if len(trajectory_points_wcs) >= FRAMES_FOR_NAIVE_ESTIMATION:
                    completed_shot_data = {
                        "shot_id": shot_count,
                        "trajectory_wcs": list(trajectory_points_wcs),
                        "naive_speed_wcs": estimated_speed_mps_wcs,
                        "naive_elev_wcs": estimated_angle_deg_elev_wcs,
                        "naive_azim_wcs": estimated_angle_deg_azim_wcs,
                        "release_frame": release_frame_info_dict['frame_no']
                    }
                    all_shots_data.append(completed_shot_data)
                    print(f"Stored data for Shot {shot_count}.")

                # *** Release the VideoWriter for this specific shot clip ***
                if shot_video_writer is not None:
                    shot_video_writer.release()
                    shot_video_writer = None
                    print(f"Finalized video clip for Shot {shot_count}.")

                release_detected_in_shot = False
                kf_initialized = False
                missed_detections_count_kf = 0

        # 4. Annotation and Display
        draw_annotations_on_frame_ccs(
            annotated_frame, frame_count, ball_bbox, ball_center_px, wrist_pos_px,
            release_detected_in_shot, release_frame_info_dict,
            trajectory_points_kf_pixels_undistorted, trajectory_points_raw_pixels_undistorted,
            shot_count, estimated_speed_mps_wcs, estimated_angle_deg_elev_wcs,  # Pass naive WCS estimates
            low_conf_flag, kf_current_predicted_pixel_point_for_drawing
        )

        # *** Write frame to the current shot's video clip if active ***
        if release_detected_in_shot and shot_video_writer is not None:
            shot_video_writer.write(annotated_frame)

        cv2.imshow("WCS KF Tracking with Release Model", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # --- End of Loop ---
    cap_g.release()
    cv2.destroyAllWindows()

    #Final Analysis
    fitting_wcs_tracking(all_shots_data, video_fps_g, session_output_dir)


