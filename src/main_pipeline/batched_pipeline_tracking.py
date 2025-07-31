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
from src.models.release_model.model_def import ReleaseRelationNet  # Import model definition

# INTRINSIC PARAMETERS
# --- Global Constants and Configuration HERO7 ---
K_global = np.array(
    [[1006.894800376467, 0.0, 964.3578313551827], [0.0, 994.7402914201482, 537.3734258659648], [0.0, 0.0, 1.0]])
D_global = np.array([0.36097005855002745, -0.013703354643132513, 0.45553079194926527, -0.32891315019398665])  # Fisheye

fx_global = K_global[0, 0];
fy_global = K_global[1, 1];
cx_global = K_global[0, 2];
cy_global = K_global[1, 2]

#Parameters for release detection as found in the utils file
SHOOTING_WRIST_KP_INDEX = utils.SHOOTING_WRIST_KP_INDEX
MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION = utils.MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
WRIST_BALL_PROXIMITY_MARGIN = utils.WRIST_BALL_PROXIMITY_MARGIN
CONFIRMATION_FRAMES_FOR_RELEASE = utils.CONFIRMATION_FRAMES_FOR_RELEASE

SHOOTING_ARM_WRIST_IDX = utils.RIGHT_WRIST
SHOOTING_ARM_ELBOW_IDX = utils.RIGHT_ELBOW
SHOOTING_ARM_SHOULDER_IDX = utils.RIGHT_SHOULDER

SAVE_RELEASE_FRAMES = True
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# RELEASE_FRAME_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "trajectory_fitting", "wcs_release_frames") # Original
# os.makedirs(RELEASE_FRAME_OUTPUT_DIR, exist_ok=True) # This general dir might not be needed if outputs are per-batch

#Detection Confidence Parameters
PRE_RELEASE_BALL_CONF = 0.35
POST_RELEASE_BALL_CONF = 0.25

#Parameters for release estimation
MAX_PIXEL_DISPLACEMENT_POST_RELEASE = 100
FRAMES_FOR_NAIVE_ESTIMATION = 20  # Min points for fitting is FRAMES_FOR_NAIVE_ESTIMATION + 1

# Real diameter of basketball in meters
BASKETBALL_REAL_DIAMETER_M = 0.24

#Class name of the ball as found in the trained model
TARGET_BALL_CLASS_NAME = "ball"

# Example 3D World Points (common to many batches, ball points will be added per batch)
COMMON_WORLD_POINTS_LIST = [
    [0.002955, 0.081709, 0.012093],  # P2: left front
    [0.050017, 0.030884, 1.955751],  # P3: left back
    [3.297116, 0, 0.046505],  # P5: right front
    [3.110943, 1.562576, 0.639417],  # P6: table front
    [3.140699, 1.546863, 1.433986],  # P7: table back
    [0, 0, 1]  # shooter foot (roughly in line with shot)
]

# --- BATCH CONFIGURATIONS ---
# IMPORTANT: Populate this list with configurations for all your batches.
# 'manual_rvec_override' and 'manual_tvec_override' can be None if not needed.
# 'initial_tvec_guess_override' should be a reasonable guess for solvePnP.
# Live video output can be set to FALSE to accelerate calculations,
SHOW_LIVE_VIDEO_OUTPUT = False

# This config list sets up batch processing
# It takes the file name for each batch, the common world points and a known location of the ball
# during a random shot to help with the vertical distance calculations (a known weakness of this research)
# Each world point also takes a pixel point respectively.
BATCH_CONFIGS = [
    {
        "batch_number": 1, "video_filename": "batch1.mp4",
        "world_points_specific": COMMON_WORLD_POINTS_LIST + [[0.65, 1.12, 0.93]],  # BALL BATCH 1
        "image_points_specific": [[50, 1045], [479, 780], [1761, 1011], [1523, 315], [1383, 361], [296, 907],
                                  [570, 466]],  # P2,P3,P5,P6,P7,Foot,Ball
        "initial_tvec_guess_override": np.array([-1.5, -0.9, 3.8]),
        "manual_rvec_override": None,
        "manual_tvec_override": np.array([[-1.6], [0.97], [3.9]])
    },
    {
        "batch_number": 2, "video_filename": "batch2.mp4",
        "world_points_specific": COMMON_WORLD_POINTS_LIST + [[0.611321, 1.267435, 0.985766]],  # Ball batch 2
        "image_points_specific": [[43, 1061], [476, 790], [1763, 1022], [1521, 324], [1380, 371], [304, 907],
                                  [567, 414]],  # P2,P3,P5,P6,P7,Foot,Ball
        "initial_tvec_guess_override": np.array([-1.5, -0.9, 3.8]),
        "manual_rvec_override": None,
        "manual_tvec_override": np.array([[-1.6], [0.97], [3.9]])
    },
    {
        "batch_number": 3, "video_filename": "batch3.mp4",
        "world_points_specific": COMMON_WORLD_POINTS_LIST + [[0.690223, 1.208302, 0.937263]],  # ball batch 3
        "image_points_specific": [[43, 1057], [475, 786], [1761, 1014], [1516, 318], [1377, 366], [352, 875],
                                  [527, 467]],  # P2,P3,P5,P6,P7,Foot,Ball
        "initial_tvec_guess_override": np.array([-1.5, -0.9, 3.8]),
        "manual_rvec_override": None,
        "manual_tvec_override": np.array([[-1.6], [0.97], [3.9]])
    },
    {
        "batch_number": 4, "video_filename": "batch4.mp4",
        "world_points_specific": COMMON_WORLD_POINTS_LIST + [[0.568437, 1.239614, 1.045756]],  # ball batch 4
        "image_points_specific": [[44, 1059], [476, 788], [1763, 1021], [1520, 324], [1380, 372], [326, 899],
                                  [565, 432]],  # P2,P3,P5,P6,P7,Foot,Ball
        "initial_tvec_guess_override": np.array([-1.5, -0.9, 3.8]),
        "manual_rvec_override": None,
        "manual_tvec_override": np.array([[-1.6], [0.97], [3.9]])
    },
    {
        "batch_number": 5, "video_filename": "batch5.mp4",
        "world_points_specific": COMMON_WORLD_POINTS_LIST + [[0.776815, 1.314616, 0.94246]],  # ball batch 5
        "image_points_specific": [[46, 1065], [478, 792], [1766, 1021], [1519, 325], [1380, 372], [318, 907],
                                  [636, 393]],  # P2,P3,P5,P6,P7,Foot,Ball
        "initial_tvec_guess_override": np.array([-1.5, -0.9, 3.8]),
        "manual_rvec_override": None,
        "manual_tvec_override": np.array([[-1.6], [0.97], [3.9]])
    },
    {
        "batch_number": 6, "video_filename": "batch6.mp4",
        "world_points_specific": COMMON_WORLD_POINTS_LIST + [[0.526551, 1.339163, 0.962354]],  # ball batch 6
        "image_points_specific": [[43, 1063], [477, 793], [1764, 1024], [1519, 326], [1380, 374], [351, 878],
                                  [528, 401]],  # P2,P3,P5,P6,P7,Foot,Ball
        "initial_tvec_guess_override": np.array([-1.5, -0.9, 3.8]),
        "manual_rvec_override": None,
        "manual_tvec_override": np.array([[-1.6], [0.97], [3.9]])
    },
    {
        "batch_number": 7, "video_filename": "batch7.mp4",
        "world_points_specific": COMMON_WORLD_POINTS_LIST + [[0.435131, 1.398297, 0.921609]],  # ball batch 7
        "image_points_specific": [[47, 1061], [478, 790], [1766, 1018], [1523, 320], [1383, 368], [358, 876],
                                  [510, 354]],  # P2,P3,P5,P6,P7,Foot,Ball
        "initial_tvec_guess_override": np.array([-1.5, -0.9, 3.8]),
        "manual_rvec_override": None,
        "manual_tvec_override": np.array([[-1.6], [0.97], [3.9]])
    },
    {
        "batch_number": 8, "video_filename": "batch8.mp4",
        "world_points_specific": COMMON_WORLD_POINTS_LIST + [[0.669286, 1.401199, 0.952018]],  # ball batch 8
        "image_points_specific": [[48, 1055], [478, 787], [1763, 1013], [1519, 318], [1381, 364], [310, 903],
                                  [582, 366]],  # P2,P3,P5,P6,P7,Foot,Ball
        "initial_tvec_guess_override": np.array([-1.5, -0.9, 3.8]),
        "manual_rvec_override": None,
        "manual_tvec_override": np.array([[-1.6], [0.97], [3.9]])
    },
    {
        "batch_number": 9, "video_filename": "batch9.mp4",
        "world_points_specific": COMMON_WORLD_POINTS_LIST + [[0.444168, 1.478248, 0.96978]],  # ball batch 9
        "image_points_specific": [[47, 1052], [477, 783], [1761, 1011], [1517, 315], [1380, 361], [307, 903],
                                  [492, 346]],  # P2,P3,P5,P6,P7,Foot,Ball
        "initial_tvec_guess_override": np.array([-1.5, -0.9, 3.8]),
        "manual_rvec_override": None,
        "manual_tvec_override": np.array([[-1.6], [0.97], [3.9]])
    },
    {
        "batch_number": 10, "video_filename": "batch10.mp4",
        "world_points_specific": COMMON_WORLD_POINTS_LIST + [[0.4578, 1.521772, 0.977554]],  # ball batch 10
        "image_points_specific": [[40, 1052], [473, 784], [1755, 1015], [1515, 320], [1378, 366], [313, 895],
                                  [516, 318]],  # P2,P3,P5,P6,P7,Foot,Ball
        "initial_tvec_guess_override": np.array([-1.5, -0.9, 3.8]),
        "manual_rvec_override": None,
        "manual_tvec_override": np.array([[-1.6], [0.97], [3.9]])
    },
]

TARGET_BATCH_NUMBER = None  # Set to an integer (e.g., 1) to run only one batch, or None to run all

# --- Parameters for tracking and kalman filter smoothening. ---
TARGET_X_WCS_FOR_IMPACT = 2.8

#Parameter after which no new predictions are made using the KF
MAX_CONSECUTIVE_MISSES_KF = 5

#Parameter after which tracking stops for that shot (20 missed frames)
MAX_CONSECUTIVE_MISSES_KF_EXTENDED = 20

#General KF parameters
KF_MEASUREMENT_NOISE_SCALE = 0.001
KF_PROCESS_NOISE_SCALE = 300
NOISE_ACCELERATION_VARIANCE = 1536

UNDISTORT_POINTS_AFTER_DETECTION = True
DETECT_ON_FULLY_UNDISTORTED_FRAME = True

MODEL_PATH = os.path.join(BASE_DIR, "src", "weights", "best_release_model.pth")
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters for enhanced release detection
# Sets a window before and after the naive release.
PATCH_SIZE = 96
INFERENCE_WINDOW_BEFORE = 10
INFERENCE_WINDOW_AFTER = 5
SHOT_DURATION_FRAMES = 120  # How long a shot is tracked post-release
SHOT_COOLDOWN_PERIOD = 50 # No. of frames before release detection is possible again

# More parameters to prevent accidental release detection which are not a shot (like bounces on floor)
MIN_BALL_HEIGHT_FOR_SHOT_GATHER = 0.45
MIN_WRIST_HEIGHT_FOR_SHOT_GATHER = 0.55

# Globals that will be set once and used across batches if not reset or overridden
R_wcs_to_ccs_g = None
tvec_wcs_to_ccs_g = None
wcs_setup_done_g = False
release_model_g = None
id_of_ball_g = None  # Will be set once in main

# Batch-specific state variables - these MUST be reset for each batch
# Declared here to indicate they are used by functions, but managed within process_batch
session_output_dir_g = ""  # Will be set per batch
shot_count_g = 0
release_detected_in_shot_g = False
release_frame_info_dict_g = None
shot_active_timer_g = 0
shot_video_writer_g = None

trajectory_points_raw_pixels_undistorted_g = []
trajectory_points_kf_pixels_undistorted_g = []
trajectory_points_ccs_g = []
trajectory_points_wcs_g = []
all_shots_data_g = []  # Data for all shots within the CURRENT batch

Zc_at_release_g = None
kf_g = None  # Kalman filter instance
kf_initialized_g = False
missed_detections_count_kf_g = 0
kf_current_predicted_pixel_point_for_drawing_g = None

# Estimates - reset per shot within a batch (global for now so that it's easier to access, consider doing OOP later with a data object)
estimated_speed_mps_g = 0.0  # CCS Naive
estimated_angle_deg_XY_ccs_g = 0.0  # CCS Naive
estimated_angle_deg_elev_ccs_g = 0.0  # CCS Naive
estimated_speed_mps_wcs_g = 0.0  # WCS Naive
estimated_angle_deg_elev_wcs_g = 0.0  # WCS Naive
estimated_angle_deg_azim_wcs_g = 0.0  # WCS Naive
fitted_params_dict_g = None
fitting_performed_for_shot_g = 0


def _init_release_state(u_rel, v_rel, release_frame_num, bbox_at_release, K_cam_batch):
    """Internal helper to initialize KF and reset current shot's trajectories."""
    # Use _g suffix for globals that are modified here
    global Zc_at_release_g, kf_initialized_g
    global estimated_speed_mps_wcs_g, estimated_angle_deg_elev_wcs_g, estimated_angle_deg_azim_wcs_g
    global trajectory_points_raw_pixels_undistorted_g, trajectory_points_kf_pixels_undistorted_g
    global trajectory_points_ccs_g, trajectory_points_wcs_g
    global kf_g  # Access the batch's KF instance

    fx_b, fy_b, cx_b, cy_b = K_cam_batch[0, 0], K_cam_batch[1, 1], K_cam_batch[0, 2], K_cam_batch[1, 2]

    trajectory_points_raw_pixels_undistorted_g.clear()
    trajectory_points_kf_pixels_undistorted_g.clear()
    trajectory_points_ccs_g.clear()
    trajectory_points_wcs_g.clear()

    estimated_speed_mps_wcs_g, estimated_angle_deg_elev_wcs_g, estimated_angle_deg_azim_wcs_g = 0.0, 0.0, 0.0
    # Also reset CCS naive estimates if they are displayed before WCS ones are ready
    global estimated_speed_mps_g, estimated_angle_deg_elev_ccs_g
    estimated_speed_mps_g, estimated_angle_deg_elev_ccs_g = 0.0, 0.0

    app_diam_px_rel = ((bbox_at_release[2] - bbox_at_release[0]) + (bbox_at_release[3] - bbox_at_release[1])) / 2.0
    Zc_at_release_g = (BASKETBALL_REAL_DIAMETER_M * (
            fx_b + fy_b) / 2.0) / app_diam_px_rel if app_diam_px_rel > 0 else 3.0  # Default to 3 if diameter is 0
    if not (0.5 < Zc_at_release_g < 10.0):  # Sanity check for calculated Zc
        print(
            f"Warning: Calculated Zc_at_release ({Zc_at_release_g:.2f}m) is out of reasonable range (0.5-10m). Defaulting to 3m.")
        Zc_at_release_g = 3.0
    Zc_at_release_g = 3.0  # Force override after calculation for stability (MENTIONED IN PAPER****)
    print(f"Zc at release for shot: {Zc_at_release_g:.2f} m")

    kf_g.statePost = np.array([u_rel, v_rel, 0, 0], dtype=np.float32).reshape(-1, 1)
    kf_g.errorCovPost = np.eye(4, dtype=np.float32) * 50.0  # Reset error cov for new shot
    kf_initialized_g = True

    xn_rel = (u_rel - cx_b) / fx_b
    yn_rel = (v_rel - cy_b) / fy_b
    Pc_rel_ccs = np.array([xn_rel * Zc_at_release_g, yn_rel * Zc_at_release_g, Zc_at_release_g])
    Pc_rel_wcs = ccs_to_wcs(Pc_rel_ccs)  # ccs_to_wcs uses R_wcs_to_ccs_g and tvec_wcs_to_ccs_g

    trajectory_points_kf_pixels_undistorted_g.append(
        {'frame': release_frame_num, 'px_coord': (u_rel, v_rel), 'type': 'kf_model_init'})
    trajectory_points_ccs_g.append({'frame': release_frame_num, 'Pc_ccs': Pc_rel_ccs, 'px_coord_type': 'kf_model_init'})
    if Pc_rel_wcs is not None:
        trajectory_points_wcs_g.append(
            {'frame': release_frame_num, 'Pc_wcs': Pc_rel_wcs, 'px_coord_type': 'kf_model_init'})


def initiate_shot_tracking(refined_frame_num, frame_buffer, K_cam_batch, frame_width_batch, frame_height_batch,
                           video_fps_batch):
    global shot_count_g, release_detected_in_shot_g, release_frame_info_dict_g, shot_active_timer_g
    global shot_video_writer_g, session_output_dir_g  # Use batch-specific output dir

    release_frame_data = next((data for data in frame_buffer if data['frame_num'] == refined_frame_num), None)
    if release_frame_data is None:
        print(f"Error: Could not find model-refined frame {refined_frame_num} in buffer.")
        return False

    final_ball_center = release_frame_data['ball_center']
    final_ball_bbox = release_frame_data['ball_bbox']

    if final_ball_center is not None and final_ball_bbox is not None:
        shot_count_g += 1
        release_detected_in_shot_g = True
        shot_active_timer_g = SHOT_DURATION_FRAMES
        release_frame_info_dict_g = {"frame_no": refined_frame_num, "ball_bbox_detection": final_ball_bbox}

        shot_dir = os.path.join(session_output_dir_g, f"shot_{shot_count_g:03d}")
        os.makedirs(shot_dir, exist_ok=True)
        clip_output_path = os.path.join(shot_dir, f"shot_{shot_count_g:03d}_clip.avi")

        # Use batch-specific frame dimensions and FPS
        shot_video_writer_g = cv2.VideoWriter(clip_output_path, cv2.VideoWriter_fourcc(*'MJPG'), video_fps_batch,
                                              (frame_width_batch, frame_height_batch))

        if not shot_video_writer_g.isOpened():
            print(f"!!! ERROR: Could not create video writer for shot clip: {clip_output_path}")
            release_detected_in_shot_g = False  # Rollback state
            shot_count_g -= 1
            return False

        start_frame_for_clip = refined_frame_num - INFERENCE_WINDOW_BEFORE
        map1_batch, map2_batch = None, None  # Recalculate map if needed, or pass if already available
        if DETECT_ON_FULLY_UNDISTORTED_FRAME:  # Assuming K_cam_batch, D_global are correct for this
            map1_batch, map2_batch = cv2.fisheye.initUndistortRectifyMap(K_cam_batch, D_global, np.eye(3), K_cam_batch,
                                                                         (frame_width_batch, frame_height_batch),
                                                                         cv2.CV_16SC2)

        for data in frame_buffer:
            f_num = data['frame_num']
            if start_frame_for_clip <= f_num < refined_frame_num:
                pre_release_img = data['image']
                # Use batch-specific K, D, maps for preprocessing
                pre_release_annotated = preprocess_frame_ccs(pre_release_img, K_cam_batch, D_global,
                                                             map1_batch, map2_batch, DETECT_ON_FULLY_UNDISTORTED_FRAME)
                draw_annotations_on_frame_ccs(  # Pass K_cam_batch for cx,cy
                    pre_release_annotated, f_num, data['ball_bbox'], data['ball_center'], data['wrist_pos'],
                    False, None, [], [], shot_count_g, 0, 0, data['low_conf_flag'], None, K_cam_batch
                )
                shot_video_writer_g.write(pre_release_annotated)

        _init_release_state(final_ball_center[0], final_ball_center[1], refined_frame_num, final_ball_bbox, K_cam_batch)
        print(f"\n>>> SHOT {shot_count_g} INITIATED (Model-Refined Frame: {refined_frame_num}) <<<")
        return True
    return False


def load_release_model(model_path):
    global release_model_g
    print(f"Loading release refinement model from: {model_path}")
    try:
        model = ReleaseRelationNet().to(MODEL_DEVICE)  # MODEL_DEVICE is global
        model.load_state_dict(torch.load(model_path, map_location=MODEL_DEVICE))
        model.eval()
        release_model_g = model
        print("Release refinement model loaded successfully.")
    except Exception as e:
        print(f"!!! ERROR: Could not load release model: {e}")
        release_model_g = None


def crop_patch(full_frame, center_xy, patch_size_val):  # Added patch_size_val
    if center_xy is None: return None
    cx, cy = center_xy
    half_size = patch_size_val // 2
    x1, y1 = cx - half_size, cy - half_size
    x2, y2 = cx + half_size, cy + half_size
    h, w, _ = full_frame.shape
    padded_frame = cv2.copyMakeBorder(full_frame, half_size, half_size, half_size, half_size, cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])
    patch = padded_frame[y1 + half_size: y2 + half_size, x1 + half_size: x2 + half_size]
    if patch.shape[0] != patch_size_val or patch.shape[1] != patch_size_val:
        patch = cv2.resize(patch, (patch_size_val, patch_size_val), interpolation=cv2.INTER_AREA)
    return patch


def refine_release_frame_with_model(candidate_frame, frame_buffer):  # Uses global release_model_g
    if release_model_g is None:
        print("Warning: Release model not loaded. Returning heuristic frame.")
        return candidate_frame

    print(f"--- Running release refinement model around frame {candidate_frame} ---")
    start_frame = candidate_frame - INFERENCE_WINDOW_BEFORE
    end_frame = candidate_frame + INFERENCE_WINDOW_AFTER
    frame_map = {data['frame_num']: data for data in frame_buffer}
    inference_samples = []
    sorted_frames_in_window = sorted([f_num for f_num in frame_map.keys() if start_frame <= f_num <= end_frame])

    for i, f_num_t in enumerate(sorted_frames_in_window):
        if i == 0: continue
        f_num_t_minus_1 = sorted_frames_in_window[i - 1]
        data_t, data_t_minus_1 = frame_map.get(f_num_t), frame_map.get(f_num_t_minus_1)
        if data_t is None or data_t_minus_1 is None: continue

        img_t, ball_bbox_t, wrist_pos_t = data_t['image'], data_t['ball_bbox'], data_t['wrist_pos']
        img_t_minus_1, ball_bbox_t_1, wrist_pos_t_1 = data_t_minus_1['image'], data_t_minus_1['ball_bbox'], \
            data_t_minus_1['wrist_pos']

        if all(item is not None for item in [ball_bbox_t, wrist_pos_t, ball_bbox_t_1, wrist_pos_t_1]):
            bc_t = (int((ball_bbox_t[0] + ball_bbox_t[2]) / 2), int((ball_bbox_t[1] + ball_bbox_t[3]) / 2))
            bc_t_1 = (int((ball_bbox_t_1[0] + ball_bbox_t_1[2]) / 2), int((ball_bbox_t_1[1] + ball_bbox_t_1[3]) / 2))

            hp_t = crop_patch(img_t, wrist_pos_t, PATCH_SIZE)
            bp_t = crop_patch(img_t, bc_t, PATCH_SIZE)
            hp_t_1 = crop_patch(img_t_minus_1, wrist_pos_t_1, PATCH_SIZE)
            bp_t_1 = crop_patch(img_t_minus_1, bc_t_1, PATCH_SIZE)

            if all(p is not None for p in [hp_t, bp_t, hp_t_1, bp_t_1]):
                inference_samples.append(
                    {"frame_num": f_num_t, "hand_t": hp_t, "hand_t-1": hp_t_1, "ball_t": bp_t, "ball_t-1": bp_t_1})

    if not inference_samples: return candidate_frame
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    scores, frame_numbers = [], []
    with torch.no_grad():
        for sample in inference_samples:
            h_t = transform(cv2.cvtColor(sample['hand_t'], cv2.COLOR_BGR2RGB)).unsqueeze(0).to(MODEL_DEVICE)
            h_t1 = transform(cv2.cvtColor(sample['hand_t-1'], cv2.COLOR_BGR2RGB)).unsqueeze(0).to(MODEL_DEVICE)
            b_t = transform(cv2.cvtColor(sample['ball_t'], cv2.COLOR_BGR2RGB)).unsqueeze(0).to(MODEL_DEVICE)
            b_t1 = transform(cv2.cvtColor(sample['ball_t-1'], cv2.COLOR_BGR2RGB)).unsqueeze(0).to(MODEL_DEVICE)
            score = release_model_g(h_t, h_t1, b_t, b_t1).item()
            scores.append(score);
            frame_numbers.append(sample['frame_num'])
    if not scores: return candidate_frame
    scores = np.array(scores)
    score_diffs = np.diff(scores, prepend=scores[0]) if len(scores) > 1 else np.array([scores[0]])
    model_refined_frame = frame_numbers[np.argmax(score_diffs)]
    print(f"Model scores: {[f'{f}:{s:.2f}' for f, s in zip(frame_numbers, scores)]}, Diffs: {score_diffs.round(2)}")
    print(f"Heuristic: {candidate_frame}, Model refined: {model_refined_frame}")
    return model_refined_frame


#TODO Anything involving WCS should be looked at in future research
def setup_wcs(camera_matrix_batch, dist_coeffs_batch,  # Use batch specific K, D might be global if same camera
              batch_world_points, batch_image_points,
              batch_initial_tvec_guess,
              batch_manual_rvec_override,
              batch_manual_tvec_override):
    global R_wcs_to_ccs_g, tvec_wcs_to_ccs_g, wcs_setup_done_g

    dist_coeffs_for_solvepnp = None
    if not DETECT_ON_FULLY_UNDISTORTED_FRAME:  # If detecting on original, pass D
        dist_coeffs_for_solvepnp = dist_coeffs_batch

    try:
        print(f"solvePnP using initial tvec guess: {batch_initial_tvec_guess}")
        success_epnp, rvec_epnp, tvec_epnp = cv2.solvePnP(
            batch_world_points, batch_image_points, camera_matrix_batch,
            dist_coeffs_for_solvepnp,
            tvec=batch_initial_tvec_guess.reshape(3, 1) if batch_initial_tvec_guess is not None else None,
            useExtrinsicGuess=True if batch_initial_tvec_guess is not None else False,
            flags=cv2.SOLVEPNP_EPNP
        )

        rvec, tvec, success = None, None, False
        if success_epnp:
            rvec, tvec = rvec_epnp, tvec_epnp
            success = True
            print("solvePnP EPNP with guess successful.")
        else:
            print("solvePnP EPNP with guess failed, trying SQPNP without guess...")
            success_sqpnp, rvec_sqpnp, tvec_sqpnp = cv2.solvePnP(
                batch_world_points, batch_image_points, camera_matrix_batch,
                dist_coeffs_for_solvepnp,
                flags=cv2.SOLVEPNP_SQPNP
            )
            if success_sqpnp:
                rvec, tvec = rvec_sqpnp, tvec_sqpnp
                success = True
                print("solvePnP SQPNP successful.")
            else:
                print("Both EPNP with guess and SQPNP failed.")

        if success:
            # Apply manual overrides if they exist for this batch
            if batch_manual_rvec_override is not None:
                print(f"Applying manual rvec override: {batch_manual_rvec_override.flatten()}")
                rvec = batch_manual_rvec_override.reshape(3, 1)
            if batch_manual_tvec_override is not None:
                print(f"Applying manual tvec override: {batch_manual_tvec_override.flatten()}")
                tvec = batch_manual_tvec_override.reshape(3, 1)

            R_wcs_to_ccs_g, _ = cv2.Rodrigues(rvec)
            tvec_wcs_to_ccs_g = tvec.reshape(3, 1)
            wcs_setup_done_g = True

            extrinsic_params_path = os.path.join(session_output_dir_g, "wcs_extrinsics.npz")
            np.savez(extrinsic_params_path, R_wcs_to_ccs=R_wcs_to_ccs_g, tvec_wcs_to_ccs=tvec_wcs_to_ccs_g,
                     K_cam=camera_matrix_batch, D_cam=dist_coeffs_batch)  # Save K,D used for this batch's solvePnP
            print(f"Saved WCS extrinsics (R, t, K, D) to {extrinsic_params_path}")

            test_world_point_idx = -2  # Assuming ball point is second to last
            if len(batch_world_points) > abs(test_world_point_idx):  # check if enough points
                test_wp = batch_world_points[test_world_point_idx]
                proj_ip, _ = cv2.projectPoints(test_wp.reshape(1, 1, 3), rvec, tvec, camera_matrix_batch,
                                               dist_coeffs_for_solvepnp)
                print(
                    f"Sanity Check: WP {test_wp} projects to IP: {proj_ip.flatten().round(1)}. Expected IP for this point: {batch_image_points[test_world_point_idx].round(1)}")

            print("WCS setup successful. R_wcs_to_ccs and tvec_wcs_to_ccs computed.")
            print("Final rvec:\n", rvec)
            # print("Final R_wcs_to_ccs:\n", R_wcs_to_ccs_g) # Can be verbose
            print("Final tvec_wcs_to_ccs:\n", tvec_wcs_to_ccs_g)
        else:
            print("WCS setup failed (solvePnP returned False).")
            wcs_setup_done_g = False
    except Exception as e:
        print(f"Error during solvePnP: {e}")
        wcs_setup_done_g = False
    return wcs_setup_done_g


def ccs_to_wcs(Pc_ccs):  # Uses global R_g, tvec_g
    if not wcs_setup_done_g or R_wcs_to_ccs_g is None or tvec_wcs_to_ccs_g is None:
        return None
    Pc_ccs_col = Pc_ccs.reshape((3, 1))
    Pc_wcs_col = R_wcs_to_ccs_g.T @ (Pc_ccs_col - tvec_wcs_to_ccs_g)
    return Pc_wcs_col.flatten()

#Note: Kalman filter could've probably been implementing in the fitting stage instead of occurring during tracking
def setup_kalman_filter(dt_val, noise_accel_var_val):  # Use passed vals
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, dt_val, 0], [0, 1, 0, dt_val], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    q_pos, q_vel = 1.0, 1.0
    kf.processNoiseCov = np.array([
        [dt_val ** 4 / 4 * q_pos, 0, dt_val ** 3 / 2 * q_pos, 0],
        [0, dt_val ** 4 / 4 * q_pos, 0, dt_val ** 3 / 2 * q_pos],
        [dt_val ** 3 / 2 * q_pos, 0, dt_val ** 2 * q_vel, 0], [0, dt_val ** 3 / 2 * q_pos, 0, dt_val ** 2 * q_vel]
    ], dtype=np.float32) * noise_accel_var_val
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KF_MEASUREMENT_NOISE_SCALE
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 500
    return kf


def undistort_point(point_xy, K_matrix, D_coeffs_val, is_fisheye=True):  # D_coeffs_val
    if point_xy is None: return None
    dist_np = np.array([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
    # Use D_coeffs_val
    undist_np = cv2.fisheye.undistortPoints(dist_np, K_matrix, D_coeffs_val, P=K_matrix) if is_fisheye \
        else cv2.undistortPoints(dist_np, K_matrix, D_coeffs_val, P=K_matrix)
    return (int(round(undist_np[0][0][0])), int(round(undist_np[0][0][1]))) if undist_np is not None else None


def preprocess_frame_ccs(frame_orig, K_matrix, D_coeffs_val, map1_val, map2_val, do_full_undistort_val):
    if do_full_undistort_val and map1_val is not None and map2_val is not None:
        return cv2.remap(frame_orig, map1_val, map2_val, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return frame_orig


def get_processed_detections_ccs(current_frame_num_val, frame_to_detect_on_val, ball_model_obj, pose_model_obj,
                                 id_ball_class_val_passed, current_ball_conf_val, K_matrix_val, D_coeffs_val,
                                 # id_ball_class_val_passed
                                 do_undistort_points_val, is_full_frame_undistorted_already,
                                 current_kf_trajectory_pixels_val, max_pixel_disp_val, is_release_detected_currently):
    # Use id_ball_class_val_passed
    ball_results = ball_model_obj(frame_to_detect_on_val, conf=current_ball_conf_val, verbose=False,
                                  classes=[id_ball_class_val_passed])
    ball_bbox_detected = utils.get_ball_bbox(ball_results, id_ball_class_val_passed)
    ball_center_undistorted_px_final = None
    ball_bbox_for_logic_val = ball_bbox_detected
    used_low_conf_heuristic_val_flag = False

    if ball_bbox_detected is not None:
        center_x_det = (ball_bbox_detected[0] + ball_bbox_detected[2]) / 2.0
        center_y_det = (ball_bbox_detected[1] + ball_bbox_detected[3]) / 2.0
        temp_ball_center_processed_px = undistort_point((center_x_det, center_y_det), K_matrix_val, D_coeffs_val) \
            if do_undistort_points_val and not is_full_frame_undistorted_already \
            else (int(round(center_x_det)), int(round(center_y_det)))

        if temp_ball_center_processed_px is not None:
            if is_release_detected_currently and current_kf_trajectory_pixels_val:
                last_known_kf_px = current_kf_trajectory_pixels_val[-1]['px_coord']
                dist_sq = (temp_ball_center_processed_px[0] - last_known_kf_px[0]) ** 2 + (
                        temp_ball_center_processed_px[1] - last_known_kf_px[1]) ** 2
                if dist_sq <= max_pixel_disp_val ** 2:
                    ball_center_undistorted_px_final = temp_ball_center_processed_px
                    if current_ball_conf_val == POST_RELEASE_BALL_CONF: used_low_conf_heuristic_val_flag = True
                else:
                    ball_bbox_for_logic_val = None
            else:
                ball_center_undistorted_px_final = temp_ball_center_processed_px
        else:
            ball_bbox_for_logic_val = None

    pose_results_val = pose_model_obj(frame_to_detect_on_val, conf=0.35, verbose=False)
    shooter_wrist_undistorted_px_final = None
    if ball_bbox_for_logic_val is not None:
        # print(f"Frame Wrist Detection: {current_frame_num_val}") # Can be very verbose
        wrist_detected_val, _ = utils.get_likely_shooter_wrist_and_person_idx(
            pose_results_val, ball_bbox_for_logic_val, SHOOTING_ARM_WRIST_IDX,
            SHOOTING_ARM_ELBOW_IDX, SHOOTING_ARM_SHOULDER_IDX, MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
        )
        if wrist_detected_val is not None:
            shooter_wrist_undistorted_px_final = undistort_point(wrist_detected_val, K_matrix_val, D_coeffs_val) \
                if do_undistort_points_val and not is_full_frame_undistorted_already \
                else (int(wrist_detected_val[0]), int(wrist_detected_val[1]))
    return ball_center_undistorted_px_final, ball_bbox_for_logic_val, shooter_wrist_undistorted_px_final, used_low_conf_heuristic_val_flag


def find_release_candidate(frame_num_val, ball_bbox_l_val, wrist_pos_undist_px, release_buffer_list_val,
                           frame_height_batch):
    is_shot_stance = False
    if ball_bbox_l_val is not None and wrist_pos_undist_px is not None:
        ball_center_y = (ball_bbox_l_val[1] + ball_bbox_l_val[3]) / 2
        if ball_center_y < frame_height_batch * MIN_BALL_HEIGHT_FOR_SHOT_GATHER and \
                wrist_pos_undist_px[1] < frame_height_batch * MIN_WRIST_HEIGHT_FOR_SHOT_GATHER:
            is_shot_stance = True
    candidate_frame = -1
    if is_shot_stance:
        wrist_in_ball = utils.is_point_inside_bbox(wrist_pos_undist_px, ball_bbox_l_val,
                                                   margin=WRIST_BALL_PROXIMITY_MARGIN)

        # Takes the first frame from the buffer (start of wrist and ball separation) after the number of confirmation frames
        if not wrist_in_ball:
            release_buffer_list_val.append(frame_num_val)
            if len(release_buffer_list_val) > CONFIRMATION_FRAMES_FOR_RELEASE: release_buffer_list_val.pop(0)
            if len(release_buffer_list_val) == CONFIRMATION_FRAMES_FOR_RELEASE:
                candidate_frame = release_buffer_list_val[0]
                release_buffer_list_val.clear()
        else:
            release_buffer_list_val.clear()
    else:
        release_buffer_list_val.clear()
    return candidate_frame, release_buffer_list_val


def update_trajectory_and_estimate_ccs(ball_point_from_kf_px_dict, video_fps_batch, K_cam_batch):
    global estimated_speed_mps_g, estimated_angle_deg_elev_ccs_g, estimated_angle_deg_XY_ccs_g, shot_count_g
    global trajectory_points_kf_pixels_undistorted_g, trajectory_points_ccs_g, trajectory_points_wcs_g
    global estimated_speed_mps_wcs_g, estimated_angle_deg_elev_wcs_g, estimated_angle_deg_azim_wcs_g
    global Zc_at_release_g  # Use the Zc for the current shot

    fx_b, fy_b, cx_b, cy_b = K_cam_batch[0, 0], K_cam_batch[1, 1], K_cam_batch[0, 2], K_cam_batch[1, 2]

    # Assuming release_detected_in_shot_g is True when this is called
    if ball_point_from_kf_px_dict is not None and Zc_at_release_g is not None:
        frame_num = ball_point_from_kf_px_dict['frame']
        u_kf, v_kf = ball_point_from_kf_px_dict['px_coord']
        point_type = ball_point_from_kf_px_dict['type']

        if not trajectory_points_kf_pixels_undistorted_g or trajectory_points_kf_pixels_undistorted_g[-1][
            'frame'] != frame_num:
            trajectory_points_kf_pixels_undistorted_g.append(ball_point_from_kf_px_dict)

        current_Pc_ccs_kf = None
        if not trajectory_points_ccs_g or trajectory_points_ccs_g[-1]['frame'] != frame_num:
            xn_kf = (u_kf - cx_b) / fx_b;
            yn_kf = (v_kf - cy_b) / fy_b
            current_Pc_ccs_kf = np.array([xn_kf * Zc_at_release_g, yn_kf * Zc_at_release_g, Zc_at_release_g])
            trajectory_points_ccs_g.append(
                {'frame': frame_num, 'Pc_ccs': current_Pc_ccs_kf, 'px_coord_type': point_type})
            if wcs_setup_done_g and current_Pc_ccs_kf is not None:
                Pc_wcs_kf = ccs_to_wcs(current_Pc_ccs_kf)  # ccs_to_wcs uses global R,t
                if Pc_wcs_kf is not None:
                    trajectory_points_wcs_g.append(
                        {'frame': frame_num, 'Pc_wcs': Pc_wcs_kf, 'px_coord_type': point_type})

        # Naive CCS estimation
        if estimated_speed_mps_g == 0.0 and len(trajectory_points_ccs_g) >= 1 + FRAMES_FOR_NAIVE_ESTIMATION:
            p_s_ccs, p_e_ccs = trajectory_points_ccs_g[0], trajectory_points_ccs_g[FRAMES_FOR_NAIVE_ESTIMATION]
            delta_Pc = p_e_ccs['Pc_ccs'] - p_s_ccs['Pc_ccs']
            d_frames = p_e_ccs['frame'] - p_s_ccs['frame']
            if video_fps_batch > 0 and d_frames > 0:
                Vc_avg = delta_Pc / (d_frames / video_fps_batch)
                estimated_speed_mps_g = np.linalg.norm(Vc_avg)
                mag_XZ = math.sqrt(Vc_avg[0] ** 2 + Vc_avg[2] ** 2) if Vc_avg[0] ** 2 + Vc_avg[2] ** 2 > 1e-9 else 1e-3
                estimated_angle_deg_elev_ccs_g = math.degrees(math.atan2(-Vc_avg[1], mag_XZ))
                estimated_angle_deg_XY_ccs_g = math.degrees(math.atan2(Vc_avg[0], Vc_avg[2]))

        # Naive WCS estimation
        if wcs_setup_done_g and estimated_speed_mps_wcs_g == 0.0 and len(
                trajectory_points_wcs_g) >= 1 + FRAMES_FOR_NAIVE_ESTIMATION:
            p_s_wcs, p_e_wcs = trajectory_points_wcs_g[0], trajectory_points_wcs_g[FRAMES_FOR_NAIVE_ESTIMATION]
            delta_Pc_wcs = p_e_wcs['Pc_wcs'] - p_s_wcs['Pc_wcs']
            d_frames_wcs = p_e_wcs['frame'] - p_s_wcs['frame']
            if video_fps_batch > 0 and d_frames_wcs > 0:
                Vc_avg_wcs = delta_Pc_wcs / (d_frames_wcs / video_fps_batch)
                estimated_speed_mps_wcs_g = np.linalg.norm(Vc_avg_wcs)
                mag_XZ_wcs = math.sqrt(Vc_avg_wcs[0] ** 2 + Vc_avg_wcs[2] ** 2) if Vc_avg_wcs[0] ** 2 + Vc_avg_wcs[
                    2] ** 2 > 1e-9 else 1e-3
                estimated_angle_deg_elev_wcs_g = math.degrees(math.atan2(Vc_avg_wcs[1], mag_XZ_wcs))
                estimated_angle_deg_azim_wcs_g = math.degrees(math.atan2(Vc_avg_wcs[2], Vc_avg_wcs[0]))
                print(
                    f"  NAIVE WCS (Shot {shot_count_g}): Speed={estimated_speed_mps_wcs_g:.2f}m/s, Elev={estimated_angle_deg_elev_wcs_g:.1f}deg, Azim={estimated_angle_deg_azim_wcs_g:.1f}deg")


def draw_annotations_on_frame_ccs(frame_to_annotate, current_frame_num, ball_bbox_d, raw_ball_center_undist_px,
                                  wrist_pos_undist_px, is_release_detected, release_info_dict,
                                  traj_kf_pix_list, traj_raw_pix_list,  # traj_raw not used
                                  shot_num, est_speed_ccs, est_angle_elev_ccs,
                                  # Using CCS estimates for primary display
                                  used_low_conf_flag, kf_current_prediction_px, K_cam_batch,
                                  frame_height_batch_val=None):  # Added K_cam_batch
    # Uses global fitted_params_dict_g, fitting_performed_for_shot_g, wcs_setup_done_g
    # Uses global WCS naive estimates: estimated_speed_mps_wcs_g, etc.
    global fitted_params_dict_g, fitting_performed_for_shot_g
    global estimated_speed_mps_wcs_g, estimated_angle_deg_elev_wcs_g, estimated_angle_deg_azim_wcs_g

    if ball_bbox_d is not None:
        l, t, r, b = [int(c) for c in ball_bbox_d]
        cv2.rectangle(frame_to_annotate, (l, t), (r, b), (0, 165, 255) if used_low_conf_flag else (255, 100, 100), 1)
    if raw_ball_center_undist_px:
        cv2.circle(frame_to_annotate, raw_ball_center_undist_px, 4, (255, 0, 255), -1)

    kf_pt_to_draw, kf_pt_type = None, 'unknown'
    if traj_kf_pix_list:
        kf_item = next((item for item in reversed(traj_kf_pix_list) if item['frame'] == current_frame_num), None)
        if kf_item:
            kf_pt_to_draw, kf_pt_type = kf_item['px_coord'], kf_item['type']
        elif kf_current_prediction_px and is_release_detected:
            kf_pt_to_draw, kf_pt_type = kf_current_prediction_px, 'kf_pred_only_not_in_traj'
    if kf_pt_to_draw:
        color = (0, 255, 0) if 'corr' in kf_pt_type or 'init' in kf_pt_type else \
            (255, 255, 0) if 'pred' in kf_pt_type else (255, 165, 0)
        cv2.circle(frame_to_annotate, kf_pt_to_draw, 6, color, -1)
    if wrist_pos_undist_px: cv2.circle(frame_to_annotate, tuple(wrist_pos_undist_px), 7, (0, 255, 255), -1)
    if is_release_detected and len(traj_kf_pix_list) >= 2:
        cv2.polylines(frame_to_annotate, [np.array([i['px_coord'] for i in traj_kf_pix_list], dtype=np.int32)], False,
                      (0, 255, 0), 2)

    y_offset = 30

    def put_text(text, color=(255, 255, 255)):
        nonlocal y_offset
        cv2.putText(frame_to_annotate, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,
                    2 if color != (255, 255, 255) else 1)
        y_offset += 25

    if is_release_detected:
        if release_info_dict: put_text(f"S{shot_num} Rls:Fr{release_info_dict['frame_no']}", (0, 0, 255))
        if traj_kf_pix_list and traj_kf_pix_list[0]['px_coord']:
            cv2.circle(frame_to_annotate, traj_kf_pix_list[0]['px_coord'], 15, (0, 0, 255), 2)

        # Display Naive WCS if available
        if wcs_setup_done_g and estimated_speed_mps_wcs_g > 0:
            put_text(
                f"N-WCS: S={estimated_speed_mps_wcs_g:.2f} E={estimated_angle_deg_elev_wcs_g:.1f} A={estimated_angle_deg_azim_wcs_g:.1f}",
                (50, 220, 100))

        # Display Fitted WCS if available FOR THE CURRENT SHOT
        if fitted_params_dict_g and fitted_params_dict_g.get("success") and fitting_performed_for_shot_g == shot_num:
            s, e, a = fitted_params_dict_g['speed_mps_wcs'], fitted_params_dict_g['angle_deg_elev_wcs'], \
                fitted_params_dict_g['angle_deg_azim_wcs']
            put_text(f"Fit-WCS: S={s:.2f} E={e:.1f} A={a:.1f}", (255, 255, 0))
    else:
        put_text("Awaiting Release")
        if not wcs_setup_done_g:
            frame_width_val = K_cam_batch[0, 2] * 2  # Approx from cx
            cv2.putText(frame_to_annotate, "WCS NOT READY", (int(frame_width_val - 200), 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

    return frame_to_annotate


def fitting_wcs_tracking(all_shots_data_batch, video_fps_batch, output_dir_batch):  # batch specific params
    global fitted_params_dict_g, fitting_performed_for_shot_g  # These reflect the LAST fitted shot

    print(f"\n--- Batch Final Analysis for {len(all_shots_data_batch)} Completed Shots ---")
    if not all_shots_data_batch: return

    for shot_data in all_shots_data_batch:
        shot_id = shot_data["shot_id"]
        shot_specific_output_dir = os.path.join(output_dir_batch, f"shot_{shot_id:03d}")
        # os.makedirs(shot_specific_output_dir, exist_ok=True) # Already created by initiate_shot_tracking

        print(f"\n--- Fitting Shot {shot_id} (Output to: {shot_specific_output_dir}) ---")

        #Note: initial guesses affect the quality of the fitted output
        if wcs_setup_done_g and len(
                shot_data["trajectory_wcs"]) >= FRAMES_FOR_NAIVE_ESTIMATION + 1:  # Use constant from tu
            initial_p0_g = shot_data["trajectory_wcs"][0]['Pc_wcs']
            initial_v_g = [shot_data["naive_speed_wcs"] if shot_data["naive_speed_wcs"] > 0 else 8.0,
                           shot_data["naive_elev_wcs"] if shot_data["naive_elev_wcs"] != 0 else 45.0,
                           shot_data["naive_azim_wcs"] if shot_data["naive_azim_wcs"] != 0 else 0.0]

            current_shot_fitted_params = tu.fit_shot_trajectory_parameters(
                shot_data["trajectory_wcs"], video_fps_batch, initial_p0_g, initial_v_g
            )
            shot_data["fitted_params"] = current_shot_fitted_params  # Store in this shot's data

            # Update global state for display on live frames if this is the current/last shot processed
            fitted_params_dict_g = current_shot_fitted_params
            fitting_performed_for_shot_g = shot_id

            wcs_traj_path = os.path.join(shot_specific_output_dir, "mono_wcs_trajectory.txt")
            with open(wcs_traj_path, 'w') as f:
                f.write("frame,x_wcs,y_wcs,z_wcs,type\n")
                for pt in shot_data["trajectory_wcs"]: f.write(
                    f"{pt['frame']},{pt['Pc_wcs'][0]},{pt['Pc_wcs'][1]},{pt['Pc_wcs'][2]},{pt['px_coord_type']}\n")
            print(f"  - Saved KF WCS trajectory for shot {shot_id}.")

            if current_shot_fitted_params and current_shot_fitted_params.get("success"):
                params_path = os.path.join(shot_specific_output_dir, "mono_fitted_params.txt")
                with open(params_path, 'w') as f_out:
                    p = current_shot_fitted_params
                    initial_p0_guess = shot_data["trajectory_wcs"][0]['Pc_wcs']
                    f_out.write(f"Fitted_P0_wcs: {p['P0_wcs'].tolist()}\n")
                    f_out.write(f"Fitted_V0_wcs_components: {p['V0_wcs_components'].tolist()}\n")
                    f_out.write(f"Fitted_Speed_mps_wcs: {p['speed_mps_wcs']}\n")
                    f_out.write(f"Fitted_Angle_Elev_wcs: {p['angle_deg_elev_wcs']}\n")
                    f_out.write(f"Fitted_Angle_Azim_wcs: {p['angle_deg_azim_wcs']}\n")
                    f_out.write(f"KB_BALL_used: {tu.KB_BALL}\n")
                    f_out.write(f"Initial_P0_Guess: {initial_p0_guess.tolist()}\n")
                    f_out.write(f"Initial_V_Angle_Guess: {initial_v_g}\n")
                print(f"  - Saved fitted parameters for shot {shot_id}.")

                # Define WCS_BB_INNER_RECT_HEIGHT, WCS_RIM_RADIUS for plotting
                # These are currently global, ensure they are appropriate or pass them
                plot_court_dims = {
                    "ft_to_rim_x": 0, "rim_y": 0,  # Dummy values if not applicable to your shotlane
                    "rim_diameter": BASKETBALL_REAL_DIAMETER_M * 1.8,  # Example, adjust
                    "backboard_x": 3.3,  # Example for table backboard X
                    "backboard_height": 1.58, "bb_inner_rect_height": 0.5  # Example
                }
                tu.plot_fitted_trajectory_wcs(
                    shot_data["trajectory_wcs"], p["P0_wcs"], p["V0_wcs_components"],
                    shot_id, video_fps_batch, shot_specific_output_dir, "fitted_trajectory_plot", plot_court_dims
                )
                print(f"  - Saved trajectory plot for shot {shot_id}.")
            else:
                print(f"  - Fitting failed for shot {shot_id}.")
        else:
            print(f"  - Not enough data or WCS not ready for shot {shot_id}. Skipping fitting.")


def process_batch(batch_config, K_cam, D_cam, base_output_dir,
                  global_ball_model, global_pose_model):
    # Use _g suffix for globals inside this function if they are modified
    # For variables that are truly local to this function's run, no suffix.
    global R_wcs_to_ccs_g, tvec_wcs_to_ccs_g, wcs_setup_done_g
    global fitted_params_dict_g, fitting_performed_for_shot_g
    global trajectory_points_raw_pixels_undistorted_g, trajectory_points_kf_pixels_undistorted_g
    global trajectory_points_ccs_g, trajectory_points_wcs_g, all_shots_data_g
    global shot_count_g, release_detected_in_shot_g, release_frame_info_dict_g, shot_active_timer_g
    global shot_video_writer_g, kf_g, kf_initialized_g, missed_detections_count_kf_g
    global kf_current_predicted_pixel_point_for_drawing_g, Zc_at_release_g
    global session_output_dir_g  # This specific global will hold the current batch's output path

    # Estimates that need to be reset per shot (which is handled by _init_release_state)
    # but also ensure they are reset at the start of a batch run before any shot.
    global estimated_speed_mps_g, estimated_angle_deg_XY_ccs_g, estimated_angle_deg_elev_ccs_g
    global estimated_speed_mps_wcs_g, estimated_angle_deg_elev_wcs_g, estimated_angle_deg_azim_wcs_g

    print(f"\n\n--- PROCESSING BATCH {batch_config['batch_number']} ---")
    print(f"Video: {batch_config['video_filename']}")

    video_path = os.path.join(VIDEO_INPUT_DIR, batch_config['video_filename'])  # VIDEO_INPUT_DIR is global
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    session_output_dir_g = os.path.join(base_output_dir, video_basename)
    os.makedirs(session_output_dir_g, exist_ok=True)

    # Reset batch-specific state variables
    frame_count_local = 0  # Local to this batch run
    shot_count_g = 0
    shot_active_timer_g = 0
    release_detected_in_shot_g = False
    release_frame_info_dict_g = None
    potential_release_buffer_local = []  # Local
    frame_buffer_local = []  # Local
    cooldown_timer_local = 0  # Local

    trajectory_points_raw_pixels_undistorted_g.clear()
    trajectory_points_kf_pixels_undistorted_g.clear()
    trajectory_points_ccs_g.clear()
    trajectory_points_wcs_g.clear()
    all_shots_data_g.clear()

    shot_video_writer_g = None
    Zc_at_release_g = None

    estimated_speed_mps_g = 0.0;
    estimated_angle_deg_XY_ccs_g = 0.0;
    estimated_angle_deg_elev_ccs_g = 0.0
    estimated_speed_mps_wcs_g = 0.0;
    estimated_angle_deg_elev_wcs_g = 0.0;
    estimated_angle_deg_azim_wcs_g = 0.0

    fitted_params_dict_g = None
    fitting_performed_for_shot_g = 0

    cap_local = cv2.VideoCapture(video_path)
    if not cap_local.isOpened(): cap_local = cv2.VideoCapture(video_path, cv2.CAP_MSMF)
    if not cap_local.isOpened():
        print(f"Error opening video for batch {batch_config['batch_number']}: {video_path}");
        return "CONTINUE"

    frame_width_batch = int(cap_local.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_batch = int(cap_local.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps_batch = cap_local.get(cv2.CAP_PROP_FPS)
    if not (0 < video_fps_batch < 1000): video_fps_batch = 30.0

    map1_batch, map2_batch = None, None
    if DETECT_ON_FULLY_UNDISTORTED_FRAME:
        map1_batch, map2_batch = cv2.fisheye.initUndistortRectifyMap(K_cam, D_cam, np.eye(3), K_cam,
                                                                     (frame_width_batch, frame_height_batch),
                                                                     cv2.CV_16SC2)

    wcs_setup_done_g = setup_wcs(K_cam, D_cam,
                                 np.array(batch_config["world_points_specific"], dtype=np.float32),  # Ensure np.array
                                 np.array(batch_config["image_points_specific"], dtype=np.float32),  # Ensure np.array
                                 batch_config["initial_tvec_guess_override"],
                                 batch_config.get("manual_rvec_override"),
                                 batch_config.get("manual_tvec_override"))
    if not wcs_setup_done_g:
        print(f"WCS setup failed for batch {batch_config['batch_number']}. Skipping.");
        cap_local.release();
        return "CONTINUE"

    kf_g = setup_kalman_filter(dt_val=1.0 / video_fps_batch, noise_accel_var_val=NOISE_ACCELERATION_VARIANCE)
    kf_initialized_g = False
    missed_detections_count_kf_g = 0
    kf_current_predicted_pixel_point_for_drawing_g = None

    window_name = f"Batch {batch_config['batch_number']} - Tracking"
    while cap_local.isOpened():
        ret, frame_original_main = cap_local.read()
        if not ret: break
        frame_count_local += 1

        current_frame_to_detect = preprocess_frame_ccs(frame_original_main, K_cam, D_cam, map1_batch, map2_batch,
                                                       DETECT_ON_FULLY_UNDISTORTED_FRAME)
        annotated_frame = current_frame_to_detect.copy()
        active_ball_conf = POST_RELEASE_BALL_CONF if release_detected_in_shot_g else PRE_RELEASE_BALL_CONF

        ball_center_px, ball_bbox, wrist_pos_px, low_conf_flag = get_processed_detections_ccs(
            frame_count_local, current_frame_to_detect, global_ball_model, global_pose_model, id_of_ball_g,
            active_ball_conf, K_cam, D_cam, UNDISTORT_POINTS_AFTER_DETECTION, DETECT_ON_FULLY_UNDISTORTED_FRAME,
            trajectory_points_kf_pixels_undistorted_g, MAX_PIXEL_DISPLACEMENT_POST_RELEASE, release_detected_in_shot_g
        )

        frame_buffer_local.append({
            "frame_num": frame_count_local, "image": frame_original_main.copy(), "ball_bbox": ball_bbox,
            "ball_center": ball_center_px, "wrist_pos": wrist_pos_px, "low_conf_flag": low_conf_flag
        })
        if len(frame_buffer_local) > (INFERENCE_WINDOW_BEFORE + INFERENCE_WINDOW_AFTER + 5): frame_buffer_local.pop(0)

        if not release_detected_in_shot_g:
            if cooldown_timer_local > 0:
                cooldown_timer_local -= 1
            else:
                candidate_frame, potential_release_buffer_local = find_release_candidate(
                    frame_count_local, ball_bbox, wrist_pos_px, potential_release_buffer_local, frame_height_batch
                )
                if candidate_frame != -1:
                    model_refined_frame = refine_release_frame_with_model(candidate_frame, frame_buffer_local)
                    if initiate_shot_tracking(model_refined_frame, frame_buffer_local, K_cam, frame_width_batch,
                                              frame_height_batch, video_fps_batch):
                        cooldown_timer_local = SHOT_COOLDOWN_PERIOD
                    else:
                        cooldown_timer_local = 30
        else:  # Post-release
            shot_active_timer_g -= 1
            ball_point_from_kf_for_update = None  # Local var for clarity
            current_ball_wcs_position = None
            if kf_initialized_g:
                predicted_kf_state = kf_g.predict()
                kf_current_predicted_pixel_point_for_drawing_g = (
                    int(round(predicted_kf_state[0, 0])), int(round(predicted_kf_state[1, 0])))
                if ball_center_px is not None:  # Valid detection
                    dist_sq = (ball_center_px[0] - kf_current_predicted_pixel_point_for_drawing_g[0]) ** 2 + \
                              (ball_center_px[1] - kf_current_predicted_pixel_point_for_drawing_g[1]) ** 2
                    if dist_sq <= MAX_PIXEL_DISPLACEMENT_POST_RELEASE ** 2:  # Close enough to KF prediction
                        measurement = np.array([[float(ball_center_px[0])], [float(ball_center_px[1])]],
                                               dtype=np.float32)
                        corrected_kf_state = kf_g.correct(measurement)
                        ball_point_from_kf_for_update = {'frame': frame_count_local, 'px_coord': (
                            int(round(corrected_kf_state[0, 0])), int(round(corrected_kf_state[1, 0]))),
                                                         'type': 'kf_corr'}
                        missed_detections_count_kf_g = 0
                    else:
                        missed_detections_count_kf_g += 1  # Detected but too far
                else:
                    missed_detections_count_kf_g += 1  # No detection

                if ball_point_from_kf_for_update is None and missed_detections_count_kf_g <= MAX_CONSECUTIVE_MISSES_KF:
                    ball_point_from_kf_for_update = {'frame': frame_count_local,
                                                     'px_coord': kf_current_predicted_pixel_point_for_drawing_g,
                                                     'type': 'kf_pred'}

                if ball_point_from_kf_for_update is not None:
                    update_trajectory_and_estimate_ccs(ball_point_from_kf_for_update, video_fps_batch, K_cam)
                    # Get ball position for termination
                    if trajectory_points_wcs_g:
                        current_ball_wcs_position = trajectory_points_wcs_g[-1]['Pc_wcs']

            shot_ended = False
            frames_since_release = frame_count_local - (
                release_frame_info_dict_g['frame_no'] if release_frame_info_dict_g else frame_count_local)
            if frames_since_release >= SHOT_DURATION_FRAMES:
                shot_ended = True;
                print(f"max duration ({SHOT_DURATION_FRAMES} frames) reached")
            # if shot_active_timer_g <= 0:
            #     shot_ended = True; print(f"Shot {shot_count_g} ended (timer).")
            elif missed_detections_count_kf_g > MAX_CONSECUTIVE_MISSES_KF_EXTENDED:  # doesnt affect kf filter
                shot_ended = True;
                print(f"Shot {shot_count_g} ended (loss). Misses: {missed_detections_count_kf_g}")
            elif current_ball_wcs_position is not None and wcs_setup_done_g:
                if current_ball_wcs_position[0] >= TARGET_X_WCS_FOR_IMPACT:
                    shot_ended = True;
                    print(f"Shot {shot_count_g}, reached Target X")

            if shot_ended:
                if len(trajectory_points_wcs_g) >= FRAMES_FOR_NAIVE_ESTIMATION + 1:  # Check enough points for fitting
                    completed_shot_data = {
                        "shot_id": shot_count_g, "trajectory_wcs": list(trajectory_points_wcs_g),
                        "naive_speed_wcs": estimated_speed_mps_wcs_g, "naive_elev_wcs": estimated_angle_deg_elev_wcs_g,
                        "naive_azim_wcs": estimated_angle_deg_azim_wcs_g,
                        "release_frame": release_frame_info_dict_g['frame_no'] if release_frame_info_dict_g else -1
                    }
                    all_shots_data_g.append(completed_shot_data)
                if shot_video_writer_g is not None: shot_video_writer_g.release(); shot_video_writer_g = None
                release_detected_in_shot_g = False;
                kf_initialized_g = False;
                missed_detections_count_kf_g = 0
                estimated_speed_mps_wcs_g, estimated_angle_deg_elev_wcs_g, estimated_angle_deg_azim_wcs_g = 0, 0, 0
                fitted_params_dict_g = None;  # Reset for next shot display

        draw_annotations_on_frame_ccs(
            annotated_frame, frame_count_local, ball_bbox, ball_center_px, wrist_pos_px,
            release_detected_in_shot_g, release_frame_info_dict_g, trajectory_points_kf_pixels_undistorted_g,
            trajectory_points_raw_pixels_undistorted_g, shot_count_g,
            estimated_speed_mps_g, estimated_angle_deg_elev_ccs_g,  # Pass Naive CCS
            low_conf_flag, kf_current_predicted_pixel_point_for_drawing_g, K_cam, frame_height_batch
        )
        if release_detected_in_shot_g and shot_video_writer_g: shot_video_writer_g.write(annotated_frame)

        if SHOW_LIVE_VIDEO_OUTPUT:  # Check the global flag
            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF  # Keep waitKey(1) for minimal delay but still process events
            if key == ord('q'):
                print(f"User pressed 'q'. Quitting current batch {batch_config['batch_number']}...")
                break
            if key == ord('b'):
                print(f"User pressed 'b'. Quitting all batch processing...")
                cap_local.release()
                if SHOW_LIVE_VIDEO_OUTPUT:  # Ensure window is destroyed if it was shown
                    cv2.destroyWindow(window_name)
                return "QUIT_ALL"
        else:  # If not showing video, add a small console progress update
            if frame_count_local % 100 == 0:  # Print progress every 100 frames
                print(f"  Batch {batch_config['batch_number']} - Processing frame: {frame_count_local}")

    cap_local.release()
    if SHOW_LIVE_VIDEO_OUTPUT:  # Only destroy window if it was created
        cv2.destroyWindow(window_name)  # Use the stored window name

    if all_shots_data_g:
        fitting_wcs_tracking(all_shots_data_g, video_fps_batch, session_output_dir_g)
    else:
        print(f"No shots fully processed for batch {batch_config['batch_number']}.")
    print(f"--- FINISHED PROCESSING BATCH {batch_config['batch_number']} ---")
    return "CONTINUE"


if __name__ == "__main__":
    # Global models loaded once
    print(os.path.join(BASE_DIR, "weights", "model2.pt"))
    ball_model_main = YOLO(model = os.path.join(BASE_DIR,"src", "weights", "model2.pt"))
    pose_model_main = YOLO(model = os.path.join(BASE_DIR,"src", "weights", 'yolov8n-pose.pt'))
    load_release_model(MODEL_PATH)  # loads into release_model_g

    id_of_ball_g = next((cid for cid, name in ball_model_main.names.items() if name.lower() == TARGET_BALL_CLASS_NAME),
                        None)
    if id_of_ball_g is None: exit(f"Class '{TARGET_BALL_CLASS_NAME}' ID not found.")

    DATA_OUTPUT_ROOT_MAIN = os.path.join(BASE_DIR, "data", "experiment_data_all_batches")
    os.makedirs(DATA_OUTPUT_ROOT_MAIN, exist_ok=True)
    VIDEO_INPUT_DIR = os.path.join(BASE_DIR,"data", "footage")  # Define VIDEO_INPUT_DIR globally

    batches_to_process = BATCH_CONFIGS
    if TARGET_BATCH_NUMBER is not None:
        selected_config = next((b for b in BATCH_CONFIGS if b["batch_number"] == TARGET_BATCH_NUMBER), None)
        if selected_config:
            batches_to_process = [selected_config]
        else:
            exit(f"Error: Target Batch {TARGET_BATCH_NUMBER} not found.")

    for config in batches_to_process:
        status = process_batch(config, K_global, D_global, DATA_OUTPUT_ROOT_MAIN,
                               ball_model_main, pose_model_main)
        if status == "QUIT_ALL": print("Quitting all batch processing by user request."); break

    if SHOW_LIVE_VIDEO_OUTPUT:  # Only destroy all windows if they might have been created
        cv2.destroyAllWindows()
    print("All specified batches processed.")
