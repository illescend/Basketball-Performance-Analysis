import os

import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
import json
# Assuming your trajectory_utils is in this path
from src.utils import trajectory_utils as tu
from scipy.signal import find_peaks, savgol_filter

os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"  # Try to disable Media Foundation
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "100"  # Try to prioritize FFMPEG

# --- Configuration Constants ---
OPTITRACK_HEADER_ROWS = 6
TIME_COL = 'Time'
BALL_COLS_MAP = {
    'Rigid Body 1 [Ball] X': 'Ball_X', 'Rigid Body 1 [Ball] Y': 'Ball_Y', 'Rigid Body 1 [Ball] Z': 'Ball_Z'
}
HAND_COLS_MAP = {
    'Rigid Body 2 [Glove] X': 'Hand_X', 'Rigid Body 2 [Glove] Y': 'Hand_Y', 'Rigid Body 2 [Glove] Z': 'Hand_Z'
}
BALL_RADIUS_M = 0.24 / 2
SHOT_COOLDOWN_S = 1
FRAMES_FOR_GT_GUESS = 10
SEPARATION_VELOCITY_THRESHOLD = 0.2
# Maximum allowed frame difference to match a GT release to a monocular detection
FRAME_MATCHING_TOLERANCE = 50

# --- Kinematic Release Detection Constants ---
GRAVITY = -9.81
# Window length for Savitzky-Golay filter. Must be odd. Larger = smoother.
SAVGOL_WINDOW = 25
SAVGOL_POLYORDER = 2 # Polynomial order for the filter.
# The release is confirmed when Y-acceleration is within this range of gravity
ACCEL_TOLERANCE_NEGATIVE = 1
ACCEL_TOLERANCE_POSITIVE = 1
# And stays within that range for this many consecutive seconds
MIN_FREEFALL_DURATION_S = 0.05
SEARCH_WINDOW_SECONDS = 0.2

# The number of rows to skip before the actual header starts.
# Based on your image, the header starts on line 3 ("Format Version...").
# The data header is on line 5 ("Frame", "Time").
# Let's target the data header on line 5.
OPTITRACK_HEADER_START_ROW = 6 # Row index (0-based) where "Frame, Time..." is.
# The number of rows the multi-level header spans (Type, Name, ID, Component).
# It seems to be 3 rows in your case: (Rigid Body), (Table), (Rotation/Position)
# Let's adjust this based on how pandas reads it. Let's try to read from row 2.
HEADER_ROW_INDICES = [2, 3, 5] # Type, Name, Component(X,Y,Z)

# The names of your Rigid Bodies as they appear in the Motive CSV file.
# This is the most important part to get right.
BALL_RB_NAME = 'bballv4'

def load_optitrack_data(csv_path):
    """
    Loads and cleans the OptiTrack CSV data with a complex multi-level header format,
    making it robust to 'Unnamed' columns for Frame and Time.
    """
    print(f"Loading new Motive CSV format from: {csv_path}")
    try:
        # Load the CSV, specifying which rows contain the multi-level header.
        # This will create a MultiIndex for the columns.
        # Corrected header rows based on CSV content analysis:
        # header[0] = row 2 (Rigid Body Name: 'bballv4', 'courtv2', 'Table')
        # header[1] = row 4 (Data Type: 'Rotation', 'Position')
        # header[2] = row 5 (Components: 'X', 'Y', 'Z', 'W', 'Frame', 'Time (Seconds)')
        df = pd.read_csv(csv_path, header=[2, 4, 5])
    except FileNotFoundError:
        print(f"ERROR: OptiTrack file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to parse CSV, likely due to header format. Error: {e}")
        return None

    # --- FIX IS HERE (the header argument in read_csv) ---
    # We will now construct the clean DataFrame piece by piece.

    df_cleaned = pd.DataFrame()

    try:
        # 1. Grab Frame and Time by their integer position (iloc).
        # This is robust to pandas naming them 'Unnamed'.
        df_cleaned['Frame'] = df.iloc[:, 0]
        df_cleaned['Time'] = df.iloc[:, 1]

        # 2. Select Ball Position data using the dynamic, multi-level name.
        # Now 'bballv4' should be correctly at level 0.
        df_cleaned['Ball_X'] = df[(BALL_RB_NAME, 'Position', 'X')]
        df_cleaned['Ball_Y'] = df[(BALL_RB_NAME, 'Position', 'Y')]
        df_cleaned['Ball_Z'] = df[(BALL_RB_NAME, 'Position', 'Z')]

    except KeyError as e:
        print(
            f"FATAL ERROR: A required column was not found. This usually means the 'BALL_RB_NAME' is incorrect or the CSV structure is unexpected.")
        print(f"  - The script failed while looking for: {e}")
        print(f"  - Your configured BALL_RB_NAME is: '{BALL_RB_NAME}'")
        # Print available names to help the user debug
        available_names = df.columns.get_level_values(0).unique()
        print(f"  - Available Rigid Body names found in file: {available_names.tolist()}")
        return None
    except IndexError:
        print(
            "FATAL ERROR: Could not access columns by position. The CSV file might be empty or have fewer than 2 columns.")
        return None

    # 3. Convert all columns to numeric and drop rows with errors.
    for col in df_cleaned.columns:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    df_cleaned.dropna(inplace=True)

    first_row = df_cleaned.iloc[0]
    print(first_row)

    print(f"Successfully loaded and cleaned {len(df_cleaned)} data points from new format.")
    return df_cleaned


def synchronize_data(optitrack_df, gopro_sync_frame, optitrack_sync_time, gopro_fps):
    """
    Calculates a global time offset based on a single sync event and adds
    synchronized GoPro time/frame columns to the OptiTrack DataFrame.
    """
    print("Synchronizing time bases...")
    if gopro_fps == 0:
        raise ValueError("GoPro FPS cannot be zero.")

    gopro_sync_time = gopro_sync_frame / gopro_fps

    # The core synchronization formula:
    # Offset = T_opti_event - T_gopro_event
    sync_offset = optitrack_sync_time - gopro_sync_time

    # Apply the offset to align OptiTrack time to the GoPro's time base
    optitrack_df['GoPro_Time_Synced'] = optitrack_df[TIME_COL] - sync_offset
    optitrack_df['GoPro_Frame_Synced'] = (optitrack_df['GoPro_Time_Synced'] * gopro_fps).round().astype(int)

    print(f"Sync event: GoPro Frame {gopro_sync_frame} <--> OptiTrack Time {optitrack_sync_time:.3f}s")
    print(f"Calculated Time Offset (T_opti - T_gopro) = {sync_offset:.4f}s")
    return optitrack_df, sync_offset


# In load_and_extract.py

def find_all_releases_kinematic(synced_optitrack_df, optitrack_fps):
    """
    Finds all shot release events using a robust heuristic that distinguishes
    between launch peaks (positive velocity) and bounce peaks (negative velocity),
    and logs the equivalent GoPro frame for each event.
    """
    print("Finding all release events using robust kinematic heuristic (launch vs. bounce)...")
    df = synced_optitrack_df.copy()
    dt = 1.0 / optitrack_fps

    # 1. Smooth position and calculate derivatives
    df['Ball_Y_smooth'] = savgol_filter(df['Ball_Y'], window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER)
    df['Velocity_Y'] = np.gradient(df['Ball_Y_smooth'], dt)
    df['Acceleration_Y'] = np.gradient(df['Velocity_Y'], dt)

    # 2. Find ALL significant positive acceleration peaks (launches AND bounces)
    min_peak_accel = 5
    min_peak_distance_frames = int(1.0 * optitrack_fps)

    candidate_peak_indices, _ = find_peaks(df['Acceleration_Y'], height=min_peak_accel,
                                           distance=min_peak_distance_frames)

    if len(candidate_peak_indices) == 0:
        print("No significant acceleration peaks found.")
        plot_kinematic_data(df, [])
        return []

    print(f"Found {len(candidate_peak_indices)} candidate acceleration peaks (launches or bounces).")

    # 3. Filter out bounce peaks by checking the velocity at the moment of the peak.
    launch_peak_indices = []
    for peak_idx in candidate_peak_indices:
        peak_time = df.loc[peak_idx, TIME_COL]
        peak_gopro_frame = df.loc[peak_idx, 'GoPro_Frame_Synced']
        peak_velocity = df.loc[peak_idx, 'Velocity_Y']

        if peak_velocity >= -0.5:
            launch_peak_indices.append(peak_idx)
        else:
            # --- MODIFIED PRINT STATEMENT ---
            print(
                f"  - Discarding peak at T={peak_time:.2f}s (GoPro Frame ~{peak_gopro_frame}) as a bounce (Velocity_Y = {peak_velocity:.2f} m/s).")

    if not launch_peak_indices:
        print("No valid launch peaks found after filtering for bounces.")
        plot_kinematic_data(df, [])
        return []

    print(f"Validated {len(launch_peak_indices)} peaks as true launches.")

    # 4. For each valid launch peak, find the precise release point that follows.
    release_timestamps = []
    for peak_idx in launch_peak_indices:
        search_end_idx = min(peak_idx + int(1.0 * optitrack_fps), len(df))
        search_window = df.iloc[peak_idx:search_end_idx]

        freefall_start = search_window[
            (search_window['Acceleration_Y'] > GRAVITY - ACCEL_TOLERANCE_NEGATIVE) &
            (search_window['Acceleration_Y'] < GRAVITY + ACCEL_TOLERANCE_POSITIVE)
            ]

        if not freefall_start.empty:
            release_idx = freefall_start.index[0]
            release_time = df.loc[release_idx, TIME_COL]

            if not release_timestamps or release_time > release_timestamps[-1] + SHOT_COOLDOWN_S:
                release_timestamps.append(release_time)
                # --- MODIFIED PRINT STATEMENT ---
                peak_time = df.loc[peak_idx, TIME_COL]
                peak_gopro_frame = df.loc[peak_idx, 'GoPro_Frame_Synced']
                release_gopro_frame = df.loc[release_idx, 'GoPro_Frame_Synced']
                print(
                    f"  - Launch peak at T={peak_time:.2f}s (GoPro Frame ~{peak_gopro_frame}) -> Confirmed release at T={release_time:.2f}s (GoPro Frame ~{release_gopro_frame})")

    print(f"Found {len(release_timestamps)} distinct and validated release events after all filtering.")

    # Generate the diagnostic plot with the final findings
    plot_kinematic_data(df, release_timestamps)

    return release_timestamps


def plot_search_window_kinematics(df_window, mono_release_time, gt_release_time, shot_id):
    """
    Creates a diagnostic plot for a single shot's search window.
    """
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Check if acceleration data exists in the dataframe
    if 'Acceleration_Y' not in df_window.columns:
        print("  - Plotting skipped: Acceleration data not available for this window.")
        return

    # Plot Y-Position on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Ball Y Position (m)', color=color, fontsize=12)
    ax1.plot(df_window[TIME_COL], df_window['Ball_Y'], color=color, label='Raw Y-Position', alpha=0.4, marker='.',
             linestyle='None')
    ax1.plot(df_window[TIME_COL], df_window['Ball_Y_smooth'], color='blue', label='Smoothed Y-Position', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for acceleration
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Y Acceleration (m/s^2)', color=color, fontsize=12)
    ax2.plot(df_window[TIME_COL], df_window['Acceleration_Y'], color=color, label='Y-Acceleration', marker='o',
             markersize=3)
    ax2.tick_params(axis='y', labelcolor=color)

    # Set a fixed, reasonable limit for the acceleration axis
    ax2.set_ylim(-20, 20)

    # Draw the "gravity zone"
    ax2.axhspan(GRAVITY - ACCEL_TOLERANCE_NEGATIVE, GRAVITY + ACCEL_TOLERANCE_POSITIVE,
                color='green', alpha=0.2, label=f'Gravity Zone')

    # Mark the monocular system's detection time
    ax1.axvline(x=mono_release_time, color='orange', linestyle='--', linewidth=2.5, label=f'Mono Detection')

    # Mark the final ground truth detection time
    if gt_release_time is not None:
        ax1.axvline(x=gt_release_time, color='purple', linestyle='-', linewidth=2.5, label=f'GT Release Found')

    # Consolidate legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    fig.tight_layout()
    plt.title(f'Release Search Window for Shot {shot_id}', fontsize=16)
    plt.show()


FREEFALL_ENTRY_THRESHOLD = -9 # m/s^2 - Adjust based on your plots (e.g., -11 or -12 is common)
STABLE_GRAVITY_LOWER_BOUND = GRAVITY - (ACCEL_TOLERANCE_NEGATIVE + 5) # e.g., -9.81 - 0.5 = -10.31
STABLE_GRAVITY_UPPER_BOUND = GRAVITY + (ACCEL_TOLERANCE_POSITIVE + 5) # e.g., -9.81 + 0.5 = -9.31
MIN_STABLE_FREEFALL_DURATION_S = 0.05 # How long it must stay in stable band

def find_release_in_window(df_window, optitrack_fps):
    """
    Searches a small DataFrame window for the first frame that marks the
    beginning of a stable free-fall period, with detailed logging for debugging.
    """
    # 1. Calculate kinematics ONLY for this small window
    dt = 1.0 / optitrack_fps
    if len(df_window) < SAVGOL_WINDOW:
        # This is a hard failure, always log it.
        print(f"  - [FAIL] Search window is too small for filtering ({len(df_window)} < {SAVGOL_WINDOW} frames).")
        return None, df_window

    df_window['Ball_Y_smooth'] = savgol_filter(df_window['Ball_Y'], window_length=SAVGOL_WINDOW,
                                               polyorder=SAVGOL_POLYORDER)
    # Using savgol_filter with deriv=2 is a more direct and stable way to get acceleration
    df_window['Acceleration_Y'] = savgol_filter(df_window['Ball_Y'],
                                                window_length=SAVGOL_WINDOW,
                                                polyorder=2,
                                                deriv=2,
                                                delta=dt)

    # 3. Validate that this is the start of a STABLE freefall period
    min_stable_frames = 30
    # A flag to ensure we only print the "no entry" message once
    found_entry_candidate = False

    for i in range(len(df_window) - min_stable_frames):
        current_idx = df_window.index[i]
        current_accel_y = df_window.loc[current_idx, 'Acceleration_Y']

        # Gate 1: Did the acceleration ever drop below the entry threshold?
        if current_accel_y < FREEFALL_ENTRY_THRESHOLD:
            found_entry_candidate = True  # We found at least one potential start point

            # Gate 2: Check for stability in the subsequent frames
            check_slice_df = df_window.iloc[i: i + min_stable_frames]
            is_stable_in_gravity_band = (
                    (check_slice_df['Acceleration_Y'] >= STABLE_GRAVITY_LOWER_BOUND) &
                    (check_slice_df['Acceleration_Y'] <= STABLE_GRAVITY_UPPER_BOUND)
            ).all()

            if not is_stable_in_gravity_band:
                # Log this failure, but only once per window to avoid clutter
                if i == 0 or df_window.loc[df_window.index[i - 1], 'Acceleration_Y'] >= FREEFALL_ENTRY_THRESHOLD:
                    print(
                        f"  - [INFO] Found potential release at T={df_window.loc[current_idx, TIME_COL]:.3f}s, but the following {min_stable_frames} frames were not stable within the gravity band.")
                continue  # Move to the next frame

            # Gate 3: Check if there's enough pre-release history to analyze
            pre_release_check_window_frames = int(0.05 * optitrack_fps)
            if pre_release_check_window_frames == 0: pre_release_check_window_frames = 1

            if i < pre_release_check_window_frames:
                print(
                    f"  - [INFO] Found stable freefall at T={df_window.loc[current_idx, TIME_COL]:.3f}s, but not enough pre-release history to validate.")
                continue  # Not enough history, skip this candidate

            # Gate 4: Check if the pre-release phase was NOT in freefall
            pre_release_slice_df = df_window.iloc[i - pre_release_check_window_frames: i]
            is_pre_release_non_gravity = (
                    (pre_release_slice_df['Acceleration_Y'] > GRAVITY + (ACCEL_TOLERANCE_POSITIVE + 1)) |
                    (pre_release_slice_df['Acceleration_Y'] < GRAVITY - (ACCEL_TOLERANCE_NEGATIVE + 1))
            ).any()

            if not is_pre_release_non_gravity:
                # This is a subtle but important failure case
                print(
                    f"  - [INFO] Found stable freefall at T={df_window.loc[current_idx, TIME_COL]:.3f}s, but the preceding frames were already in freefall (likely mid-flight, not a release).")
                continue  # This is not a true release event

            # --- SUCCESS ---
            # If we passed all gates, we found our release
            return df_window.loc[current_idx, TIME_COL], df_window

    # If the loop finishes without returning, log the final reason for failure
    if not found_entry_candidate:
        min_accel_in_window = df_window['Acceleration_Y'].min()
        print(
            f"  - [FAIL] No release found. The acceleration never dropped below the entry threshold of {FREEFALL_ENTRY_THRESHOLD:.2f} m/s^2. (Min acceleration in window was {min_accel_in_window:.2f})")
    else:
        # This case means we found candidates, but none of them passed all the subsequent checks (stability, history, etc.)
        print(
            f"  - [FAIL] No release found. Potential candidates were found, but none met the stability and pre-release conditions.")

    return None, df_window

def plot_kinematic_data(df, release_timestamps):
    """
    Creates a diagnostic plot showing Y-Position, Y-Acceleration, and detected releases.
    """
    print("Generating diagnostic plot for kinematic release detection...")

    fig, ax1 = plt.subplots(figsize=(20, 10))

    # Plot Y-Position on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Ball Y Position (m)', color=color)
    ax1.plot(df[TIME_COL], df['Ball_Y'], color=color, label='Raw Y-Position', alpha=0.5)
    ax1.plot(df[TIME_COL], df['Ball_Y_smooth'], color='blue', label='Smoothed Y-Position', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':')

    # Create a second y-axis for acceleration
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Y Acceleration (m/s^2)', color=color)
    ax2.plot(df[TIME_COL], df['Acceleration_Y'], color=color, label='Y-Acceleration')
    ax2.tick_params(axis='y', labelcolor=color)

    ax2.set_ylim(-10, 10)

    # Draw the "gravity zone"
    ax2.axhspan(GRAVITY - ACCEL_TOLERANCE_NEGATIVE, GRAVITY + ACCEL_TOLERANCE_POSITIVE,
                color='green', alpha=0.2, label='Gravity Zone (Free-fall)')

    # Mark the detected release points
    if release_timestamps:
        for t in release_timestamps:
            ax1.axvline(x=t, color='purple', linestyle='--', linewidth=2,
                        label=f'Detected Release @ {t:.2f}s' if t == release_timestamps[0] else "")

    # Consolidate legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    fig.tight_layout()
    plt.title('Kinematic Release Detection Diagnostic Plot')
    plt.show()

def calculate_gt_parameters(synced_optitrack_df, release_time, optitrack_fps):
    """
    For a single release, this function extracts the free-flight trajectory and
    calculates the initial release parameters by FITTING a physics model,
    identical to the monocular system's method.

    Args:
        synced_optitrack_df (pd.DataFrame): The full OptiTrack data.
        release_time (float): The precise timestamp of the shot release.
        optitrack_fps (float): The frame rate of the OptiTrack system.

    Returns:
        dict: A dictionary containing the trajectory and the FITTED parameters.
              Returns None if the shot is invalid.
    """
    df = synced_optitrack_df

    # 1. Find the start and end of the shot trajectory
    shot_start_index = df[df[TIME_COL] >= release_time].index[0]
    impact_df = df.loc[shot_start_index:][df.loc[shot_start_index:]['Ball_Y'] <= BALL_RADIUS_M]
    if impact_df.empty: return None
    shot_end_index = impact_df.index[0]

    trajectory_df = df.loc[shot_start_index:shot_end_index]
    if len(trajectory_df) < FRAMES_FOR_GT_GUESS: return None

    # Create the list of dictionaries with positions
    observed_wcs_positions_for_fitting = [
        {'Pc_wcs': row[['Ball_X', 'Ball_Y', 'Ball_Z']].values}
        for _, row in trajectory_df.iterrows()
    ]

    # Create a separate, correctly sorted time array from the native OptiTrack time
    first_time = trajectory_df.iloc[0][TIME_COL]
    time_points_for_fitting = (trajectory_df[TIME_COL] - first_time).values

    # 3. Generate a good INITIAL GUESS for the optimizer using the finite difference method.
    p_start_guess = observed_wcs_positions_for_fitting[0]['Pc_wcs']
    p_end_guess = observed_wcs_positions_for_fitting[FRAMES_FOR_GT_GUESS - 1]['Pc_wcs']
    t_start_guess = time_points_for_fitting[0]
    t_end_guess = time_points_for_fitting[FRAMES_FOR_GT_GUESS - 1]

    v_guess_components = (p_end_guess - p_start_guess) / (t_end_guess - t_start_guess)
    vx_g, vy_g, vz_g = v_guess_components

    speed_guess = np.linalg.norm(v_guess_components)
    elev_guess = np.degrees(np.arctan2(vy_g, np.sqrt(vx_g ** 2 + vz_g ** 2)))
    azim_guess = np.degrees(np.arctan2(vz_g, vx_g))

    initial_v_speed_angle_guess = [speed_guess, elev_guess, azim_guess]

    observed_wcs_data_for_fitting = []
    for i, row in trajectory_df.iterrows():
        observed_wcs_data_for_fitting.append({
            'frame': i,  # Use the index as a fake, unique frame number
            'Pc_wcs': row[['Ball_X', 'Ball_Y', 'Ball_Z']].values
        })

    fitted_params_result = tu.fit_shot_trajectory_parameters(
        observed_wcs_trajectory_data=observed_wcs_data_for_fitting,
        fps=optitrack_fps,  # We use the OptiTrack FPS here
        initial_P0_guess=p_start_guess,
        initial_V_speed_angle_guess=initial_v_speed_angle_guess
    )

    if not fitted_params_result or not fitted_params_result.get("success"):
        print("  - WARNING: Ground truth fitting failed. Skipping shot.")
        return None

    # Add the original trajectory DataFrame to the results for saving
    fitted_params_result["trajectory_df"] = trajectory_df
    return fitted_params_result


# This where you enter your sync frames between the gopro and optitrack system
# This experiment used a ball drop as a sync event.
# GoPro frame: frame where the ball hits the ground and folds in on itself (its lowest point)
# Optitrack Time: the time with lowest y position of the ball as found in the csv
batch_sync_data = {
    1: {'gopro_frame': 567, 'optitrack_time': 5.416667},
    2: {'gopro_frame': 528, 'optitrack_time': 4.970833},
    3: {'gopro_frame': 532, 'optitrack_time': 5.058333},
    4: {'gopro_frame': 496, 'optitrack_time': 4.845833},
    5: {'gopro_frame': 519, 'optitrack_time': 5.041667},
    6: {'gopro_frame': 553, 'optitrack_time': 5.35},
    7: {'gopro_frame': 526, 'optitrack_time': 5.016667},
    8: {'gopro_frame': 551, 'optitrack_time': 5.0},
    9: {'gopro_frame': 497, 'optitrack_time': 4.795833},
    10: {'gopro_frame': 537, 'optitrack_time': 5.025},
}

# --- Main Driver Function ---
if __name__ == '__main__':
    # --- 1. DEFINE SYNC EVENT PARAMETERS (THE ONLY MANUAL STEP) ---
    # Find these values manually from your data as described in the workflow.

    SELECTED_BATCH_NUMBER = 10

    GOPRO_SYNC_FRAME = batch_sync_data[SELECTED_BATCH_NUMBER]['gopro_frame']
    OPTITRACK_SYNC_TIME = batch_sync_data[SELECTED_BATCH_NUMBER]['optitrack_time']

    # --- 2. SETUP PATHS AND GET VIDEO METADATA ---
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    VIDEO_INPUT_DIR = os.path.join(BASE_DIR,"data", "footage")
    video_path = os.path.join(VIDEO_INPUT_DIR, f"batch{SELECTED_BATCH_NUMBER}.mp4")
    SESSION_NAME = os.path.splitext(os.path.basename(video_path))[0]

    cap_g = cv2.VideoCapture(video_path)
    if not cap_g.isOpened():
        print(f"FATAL ERROR: Could not open video file at {video_path}")
        exit()
    GOPRO_FPS = cap_g.get(cv2.CAP_PROP_FPS)
    cap_g.release()
    if GOPRO_FPS == 0:
        raise ValueError("Video FPS is 0. Check video file.")

    SESSION_DIR = os.path.join(BASE_DIR, "data","footage","monocular_experiment_data", SESSION_NAME)
    OPTITRACK_CSV_PATH = os.path.join(SESSION_DIR, f"batch_shots_10_{SELECTED_BATCH_NUMBER}.csv")
    # Session directory should have the optitrack data

    # --- 3. PIPELINE EXECUTION ---

    # Load data
    opti_df = load_optitrack_data(OPTITRACK_CSV_PATH)
    if opti_df is None: exit()

    optitrack_fps = 1.0 / opti_df[TIME_COL].diff().mean()
    print(f"Detected OptiTrack FPS: {optitrack_fps:.2f}")

    # Synchronize data using the manually found sync points
    synced_df, time_offset = synchronize_data(opti_df, GOPRO_SYNC_FRAME, OPTITRACK_SYNC_TIME, GOPRO_FPS)

    shot_folders = sorted([d for d in os.listdir(SESSION_DIR) if d.startswith("shot_")])
    print(f"\nFound {len(shot_folders)} shot folders created by the monocular system. Finding GT matches...")

    populated_count = 0
    for shot_folder_name in shot_folders:
        shot_dir = os.path.join(SESSION_DIR, shot_folder_name)
        shot_id = int(shot_folder_name.split('_')[-1])
        print(f"\n--- Processing {shot_folder_name} ---")

        try:
            # a. Get the monocular system's detected release frame
            mono_traj_df = pd.read_csv(os.path.join(shot_dir, "mono_wcs_trajectory.txt"))
            mono_release_frame = mono_traj_df['frame'].iloc[0]

            # b. Convert it to the OptiTrack time base
            mono_release_time_equiv = (mono_release_frame / GOPRO_FPS) + time_offset

            # c. Define the search window in the OptiTrack data
            start_search_time = mono_release_time_equiv - SEARCH_WINDOW_SECONDS
            end_search_time = mono_release_time_equiv + SEARCH_WINDOW_SECONDS
            search_window_df = synced_df[
                (synced_df[TIME_COL] >= start_search_time) & (synced_df[TIME_COL] <= end_search_time)].copy()

            print(
                f"  - Mono release frame: {mono_release_frame}. Searching for GT release around T={mono_release_time_equiv:.3f}s...")

            # d. Find the precise GT release within this window
            # The function now also returns the processed dataframe for plotting
            gt_release_time, processed_window_df = find_release_in_window(search_window_df, optitrack_fps)

            # --- PLOTTING CALL ---
            # Plot the search window kinematics regardless of whether a release was found
            plot_search_window_kinematics(processed_window_df, mono_release_time_equiv, gt_release_time, shot_id)

            if gt_release_time is None:
                print("  - FAILED: Could not find a valid GT release in the search window.")
                continue

            print(f"  - SUCCESS: Found GT release at T={gt_release_time:.3f}s.")

            # e. Popoulate the folder with the GT data
            gt_results = calculate_gt_parameters(synced_df, gt_release_time, optitrack_fps)
            if gt_results is None:
                print(f"  - FAILED: GT data for this match was invalid or fitting failed.")
                continue

            gt_traj_path = os.path.join(shot_dir, "gt_wcs_trajectory.csv")
            gt_results["trajectory_df"].to_csv(gt_traj_path, index=False)

            gt_gopro_frame_equiv = int(round((gt_release_time - time_offset) * GOPRO_FPS))
            summary_data = {
                "T_OptiTrackRelease": gt_release_time, "GoPro_Frame_Equivalent": gt_gopro_frame_equiv,
                "P0_wcs_fitted": gt_results["P0_wcs"].tolist(),
                "V0_wcs_components_fitted": gt_results["V0_wcs_components"].tolist(),
                "Speed_mps_wcs_fitted": gt_results["speed_mps_wcs"],
                "Angle_Elev_wcs_fitted": gt_results["angle_deg_elev_wcs"],
                "Angle_Azim_wcs_fitted": gt_results["angle_deg_azim_wcs"]
            }
            gt_summary_path = os.path.join(shot_dir, "gt_summary.json")
            with open(gt_summary_path, 'w') as f:
                json.dump(summary_data, f, indent=4)

            print(f"  - Successfully populated folder with GT data.")
            populated_count += 1

        except (FileNotFoundError, IndexError) as e:
            print(f"  - FAILED: Could not process folder. Error: {e}")