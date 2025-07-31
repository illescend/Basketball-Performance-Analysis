# Basketball-Performance-Analysis

This repository contains the code and methodology for a computer vision system designed to extract and analyze 3D basketball shot kinematics from a single camera.

## Project Structure

- **/data/**: Contains input footage and ground truth data. (Note: Due to size, full datasets are not included in the repo).
- **/src/**: All Python source code.
  - **/analysis/**: Script for analyzing and visualizing results.
  - **/data_processing/**: Script for processing ground truth data (e.g., from OptiTrack).
  - **/main_pipeline/**: The core script for running the monocular analysis pipeline (batch vs single).
  - **/models/**: Definitions and weights for custom-trained models (e.g., ReleaseRelationNet).
  - **/utils/**: Shared utility functions.
  - **/weights/**: Pre-trained third-party model weights (e.g., YOLO).

## Setup

1.  Clone the repository.
2.  Create a Python virtual environment (e.g., `python -m venv venv`).
3.  Activate the environment (e.g., `source venv/bin/activate` or `.\venv\Scripts\activate`).
4.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
5.  Download the necessary YOLO weights (from this repo or other customs models) and place them in the `/weights/` directory.
6.  Place input videos in `/data/footage/`.

## How to Run

### 1. Run the Main Analysis Pipeline
This script processes the videos to generate trajectories and fitted parameters.

### File/Directory Descriptions and Key Parameters

This section describes the function of the main Python scripts and highlights important parameters that may need to be adjusted for different datasets or future research.

---

#### `src/main_pipeline/batch_pipeline_tracking.py`

This is the core script of the monocular analysis pipeline. It takes raw video files as input and performs the end-to-end process of detecting shots, tracking the ball, reconstructing the 3D trajectory in a World Coordinate System (WCS), and fitting a physics-based model to estimate initial release parameters.

**Functionality:**
1.  **Batch Processing:** It is designed to loop through a list of pre-configured "batches," where each batch corresponds to a specific video and its associated calibration data.
2.  **State Machine:** For each frame, it operates in a state machine: either searching for a new shot release or tracking an active shot.
3.  **Object/Pose Detection:** Utilizes YOLO models to detect the basketball and the shooter's pose in each frame.
4.  **Release Detection:** Employs a two-stage process: a fast heuristic based on hand-ball separation to identify a candidate release, followed by a custom CNN (`ReleaseRelationNet`) to refine the precise release frame.
5.  **3D Reconstruction:**
    *   Establishes a WCS for each batch using OpenCV's `solvePnP` with pre-defined 2D-3D point correspondences.
    *   Tracks the ball in 2D using a Kalman Filter.
    *   Converts the 2D trajectory to a 3D trajectory in Camera Coordinates (CCS) using a fixed-depth assumption, and then transforms it to the WCS.
6.  **Parameter Fitting:** Fits a physics-based model (including air drag) to the reconstructed 3D WCS trajectory to find the initial release parameters (speed, elevation, azimuth).
7.  **Output Generation:** For each detected shot, it saves an annotated video clip, the reconstructed WCS trajectory (`mono_wcs_trajectory.txt`), and the fitted parameters (`mono_fitted_params.txt`). It also saves the camera's extrinsic parameters (`wcs_extrinsics.npz`) for each batch.

**Key Parameters & Points of Attention:**
*   `TARGET_BATCH_NUMBER`: Set to an integer to run a single specific batch for debugging/testing, or `None` to run all batches defined in `BATCH_CONFIGS`.
*   `BATCH_CONFIGS`: **(CRITICAL)** This is the main configuration list. For any new dataset, you must add a new configuration dictionary containing:
    *   `video_filename`: The name of the new video file.
    *   `world_points_specific` & `image_points_specific`: The manually measured 3D world coordinates and their corresponding 2D pixel coordinates from a reference frame in the new video. The accuracy of the entire WCS reconstruction depends entirely on the quality of these points.
    *   `manual_rvec_override` & `manual_tvec_override`: These can be set to manually override the `solvePnP` output for camera pose if the automated result is unsatisfactory. This is a key area for manual tuning if WCS alignment is poor.
*   `K_global`, `D_global`: The camera intrinsic matrix and distortion coefficients. These **must** be re-calibrated and updated if a different camera is used.
*   `Zc_at_release`: The assumed depth of the ball from the camera at release. The code defaults to a stable value of 3.0 meters. This is a major simplifying assumption and a primary source of 3D reconstruction error. It may need to be adjusted for setups with significantly different camera-to-subject distances.

---

#### `src/data_processing/load_and_extract_gt.py`

This script processes the high-fidelity ground truth (GT) data from the OptiTrack system. Its purpose is to take the raw OptiTrack output, identify the release events kinematically, and generate clean GT files for each shot folder created by the main pipeline.

**Key Parameters & Points of Attention:**
*   `SELECTED_BATCH_NUMBER`: Specifies which batch of OptiTrack data to process.
*   `batch_sync_data`: A dictionary containing the manually identified synchronization points (GoPro frame and OptiTrack time) for each batch. This **must** be updated for new recordings.
*   `SAVGOL_WINDOW`, `SAVGOL_POLYORDER`: Parameters for the Savitzky-Golay smoothing filter. These were tuned for the current dataset and may need adjustment if new OptiTrack data has different noise characteristics.
*   `ACCEL_TOLERANCE_NEGATIVE/POSITIVE`, `MIN_FREEFALL_DURATION_S`: Thresholds for the kinematic release detection heuristic. These are sensitive parameters that define what the system considers "stable free fall."

---

#### `src/analysis/analyze_and_visualize.py`

This script is the final stage of the project. It takes the outputs from both the monocular pipeline and the GT processing script, performs a detailed comparison, and generates all the summary statistics, tables, and plots for the paper.

**Key Parameters & Points of Attention:**
*   `EXPERIMENT_DATA_ROOT`: Must point to the root directory where the batch folders (containing `shot_XXX` subfolders) are located.
*   `EXCLUSION_LIST`: A dictionary used to manually exclude specific shots that were identified as problematic (e.g., bad GT data, clear tracking failure). This may need to be updated when analyzing new data.
*   `MAX_X_WCS_FOR_ANALYSIS`: Defines the endpoint for trajectory comparison to ensure consistency. This value is tied to the WCS setup and should represent a plane just before the target (e.g., backboard).

#### `src/utils/general_utils.py`

This script is a collection of helper functions primarily used for computer vision geometry and processing the outputs from the YOLO object and pose detection models.

**Functionality:**
*   **Bounding Box & Keypoint Extraction:** Contains functions like `get_ball_bbox` to parse the raw output from the YOLOv8 model and extract the primary basketball bounding box.
*   **Shooter Identification:** The core function `get_likely_shooter_wrist_and_person_idx` identifies the most probable shooter from multiple detected people in a frame. It does this by evaluating keypoint confidences (wrist, elbow, shoulder) and enforcing geometric constraints (e.g., wrist-to-ball proximity, reasonable arm extension angle) to ensure a plausible shooting posture.
*   **Geometric Calculations:** Provides basic geometric helpers like `is_point_inside_bbox`, `calculate_distance`, and `calculate_angle` that are used throughout the pipeline for heuristic checks.

**Key Parameters & Points of Attention:**
*   **Keypoint Indices (`RIGHT_WRIST`, `RIGHT_ELBOW`, etc.):** These integer constants are tied to the **COCO keypoint format**, which is the output standard for the `yolov8n-pose.pt` model. If a different pose estimation model with a different keypoint mapping were used, these indices would need to be updated.
*   **Confidence & Geometric Thresholds:** Constants like `MIN_WRIST_CONFIDENCE`, `MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION`, and `MIN_SHOULDER_ELBOW_WRIST_ANGLE_DEG` are crucial tuning parameters. They define the heuristics for identifying a valid shooter and may need to be adjusted for videos with different lighting, camera angles, or resolutions to maintain robustness.

---

#### `src/utils/trajectory_utils.py`

This script contains the core physics-based logic of the project. It handles the simulation and optimization (fitting) of 3D basketball trajectories in the World Coordinate System (WCS).

**Functionality:**
*   **Trajectory Fitting (`fit_shot_trajectory_parameters`):** This is the main function of the module. It takes a sequence of observed 3D WCS points (from the monocular or GT system) as input. It then uses an optimization algorithm (`scipy.optimize.minimize` with L-BFGS-B) to find the six initial release parameters ($P_0 = [X_0, Y_0, Z_0]$, speed, elevation, and azimuth) that best explain the observed trajectory according to a physics model that includes gravity and quadratic air drag.
*   **Trajectory Simulation (`simulate_projectile_with_drag`):** This is the forward model used by the fitter. It takes a set of initial release conditions and solves a system of ordinary differential equations (ODEs) to predict the ball's 3D path over time.
*   **Visualization (`plot_fitted_trajectory_wcs`):** A utility function to generate plots comparing an observed WCS trajectory to the trajectory simulated from the fitted parameters. This is essential for visual debugging and for creating figures for analysis.

**Key Parameters & Points of Attention:**
*   **Physics Constants (`GRAVITY_ACCELERATION`, `KB_BALL`):** These constants define the physical model. `KB_BALL` is the derived air drag parameter, which depends on the mass, radius, and drag coefficient of a standard basketball. For analyzing other objects, these values would need to be changed.
*   **Optimization Bounds:** Inside `fit_shot_trajectory_parameters`, there are defined `bounds` for the optimization (e.g., speed between 4-15 m/s, elevation between 15-75 degrees). These constraints help the fitter converge to physically plausible solutions and prevent it from finding unrealistic local minima. They may need to be adjusted for different types of shots (e.g., very short layups vs. long-range shots).

---

#### `src/models/release_model/` (ReleaseRelationNet)

This directory contains the custom-trained machine learning model responsible for refining the precise moment of shot release. It is a more sophisticated approach than the simple geometric heuristic.

**Functionality & Concept:**
The purpose of `ReleaseRelationNet` is to act as a binary classifier that distinguishes between a "pre-release" state (hand is in control of the ball) and a "post-release" state (ball is in free flight).

*   **Input:** The model takes a unique spatio-temporal input consisting of four `96x96` pixel image patches from two consecutive frames, `t-1` and `t`: (`hand_t`, `hand_t-1`, `ball_t`, `ball_t-1`). This allows the model to analyze not just the static appearance but also the motion and change in relationship between the hand and the ball.
*   **Architecture:** It uses a Siamese-like or Two-Stream architecture. Two separate CNN branches process the hand and ball patch sequences to extract high-level feature vectors. These feature vectors are then concatenated and fed into a "relation head" (a series of fully-connected layers) that learns to interpret the relationship between the hand's motion features and the ball's motion features.
*   **Output:** The model outputs a single "release score" between 0.0 (strong confidence of pre-release) and 1.0 (strong confidence of post-release) for the transition from frame `t-1` to `t`.
*   **Final Decision:** The system identifies the final release frame by finding the largest positive *increase* in this score within a search window. This sharp jump in the score signifies the most likely moment the ball left the shooter's hand.

**Key Files & Points of Attention:**
*   **`model_def.py`:** Contains the PyTorch code (`nn.Module`) that defines the CNN architecture described above.
*   **`best_release_model.pth`:** The file containing the saved weights of the trained model.
*   **Training Data Dependency:** The model's performance is entirely dependent on the quality and quantity of the manually labeled data used to train it. Its generalizability to different players, lighting conditions, and camera angles is a function of the diversity of its training set.
*   **`PATCH_SIZE`:** The `96x96` patch size is a fixed hyperparameter. Any changes to this would require re-training the model from scratch.