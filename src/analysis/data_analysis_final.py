import os
import json
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import cv2  # Needed for cv2.projectPoints

# --- NEW DEPENDENCY: Import trajectory_utils for re-fitting ---
from src.utils import trajectory_utils as tu

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXPERIMENT_DATA_ROOT = os.path.join(BASE_DIR, "data","monocular_experiment_data")

EXCLUSION_LIST = {
    'batch1': [6, 8], 'batch2': [2, 3], 'batch3': [4, 5], 'batch4': [3, 9, 10],
    'batch5': [4, 5, 11, 12], 'batch6': [4, 6, 8, 9], 'batch7': [3, 4, 7, 9, 12],
    'batch8': [5, 6, 10], 'batch9': [5, 6, 8, 9, 12], 'batch10': [2, 3, 6, 7, 9, 10, 11]
}

GOPRO_FPS = 119.88
# Based on court dimensions of experiment setup.
# Helps to clip data before impact
MAX_X_WCS_FOR_ANALYSIS = 2.8  # meters


def load_batch_extrinsics(batch_dir):
    extrinsic_path = os.path.join(batch_dir, "wcs_extrinsics.npz")
    if not os.path.exists(extrinsic_path):
        print(f"  - WARNING: wcs_extrinsics.npz not found in {batch_dir}.")
        return None, None, None, None
    try:
        data = np.load(extrinsic_path)
        return data['K_cam'], data['D_cam'], data['R_wcs_to_ccs'], data['tvec_wcs_to_ccs']
    except Exception as e:
        print(f"  - ERROR loading wcs_extrinsics.npz from {batch_dir}: {e}")
        return None, None, None, None


def project_wcs_to_pixels(wcs_points_df, K_cam, D_cam, R_wcs_to_ccs, tvec_wcs_to_ccs, col_x='Ball_X', col_y='Ball_Y',
                          col_z='Ball_Z'):
    if wcs_points_df.empty or K_cam is None: return pd.DataFrame(columns=['u_proj', 'v_proj'])
    wcs_coords = wcs_points_df[[col_x, col_y, col_z]].values
    projected_pixels_list = []
    for p_wcs in wcs_coords:
        p_ccs = (R_wcs_to_ccs @ p_wcs.reshape(3, 1)) + tvec_wcs_to_ccs
        rvec_ident, _ = cv2.Rodrigues(np.eye(3))
        tvec_zeros = np.zeros((3, 1))
        pixel_coords, _ = cv2.projectPoints(p_ccs.reshape(1, 1, 3), rvec_ident, tvec_zeros, K_cam, D_cam)
        projected_pixels_list.append(pixel_coords[0][0])
    return pd.DataFrame(projected_pixels_list, columns=['u_proj', 'v_proj'], index=wcs_points_df.index)


def load_shot_data(shot_dir):
    try:
        with open(os.path.join(shot_dir, "gt_summary.json"), 'r') as f:
            gt_summary = json.load(f)
        mono_traj_wcs = pd.read_csv(os.path.join(shot_dir, "mono_wcs_trajectory.txt"))
        if mono_traj_wcs.empty: raise ValueError("Mono traj empty")
        with open(os.path.join(shot_dir, "mono_fitted_params.txt"), 'r') as f:
            mono_params_text = f.readlines()
        mono_summary = {}
        for line in mono_params_text:
            if ":" not in line: continue
            key, val = [x.strip() for x in line.strip().split(':', 1)]
            if key == "Fitted_Speed_mps_wcs":
                mono_summary['Speed_mps_wcs_fitted'] = float(val)
            elif key == "Fitted_Angle_Elev_wcs":
                mono_summary['Angle_Elev_wcs_fitted'] = float(val)
            elif key == "Fitted_Angle_Azim_wcs":
                mono_summary['Angle_Azim_wcs_fitted'] = float(val)
            elif key == "Fitted_P0_wcs":
                mono_summary['P0_wcs_fitted'] = json.loads(val)
        mono_summary['GoPro_Frame_Detected'] = int(mono_traj_wcs.iloc[0]['frame'])
        mono_summary['P0_wcs_detected_from_traj_start'] = mono_traj_wcs.iloc[0][['x_wcs', 'y_wcs', 'z_wcs']].tolist()
        gt_traj = pd.read_csv(os.path.join(shot_dir, "gt_wcs_trajectory.csv"))
        if gt_traj.empty: raise ValueError("GT traj empty")
        return gt_summary, mono_summary, gt_traj, mono_traj_wcs
    except Exception as e:
        print(f"  - ERROR loading data for {os.path.basename(shot_dir)}: {e}")
        return None, None, None, None


def truncate_segment_at_x_wcs(segment_df, x_wcs_col_name, max_x_val):
    if segment_df.empty: return segment_df
    idx_before_max_x = segment_df[x_wcs_col_name] < max_x_val
    if not idx_before_max_x.any():
        return segment_df.iloc[:1] if segment_df[x_wcs_col_name].iloc[0] >= max_x_val else pd.DataFrame(
            columns=segment_df.columns)
    indices_at_or_after_max_x = np.where(segment_df[x_wcs_col_name] >= max_x_val)[0]
    return segment_df.iloc[:indices_at_or_after_max_x[0] + 1] if len(indices_at_or_after_max_x) > 0 else segment_df


# --- MODIFIED FUNCTION ---
def calculate_aligned_rmse(gt_segment_aligned, mono_segment_aligned):
    """Calculates RMSE and Max Deviation on two segments that are already aligned and truncated."""
    gt_total_flight_duration_s = 0.0
    if not gt_segment_aligned.empty and len(gt_segment_aligned) > 1:
        gt_total_flight_duration_s = gt_segment_aligned['Time'].max() - gt_segment_aligned['Time'].min()

    if mono_segment_aligned.empty or gt_segment_aligned.empty:
        # Return four values to match the expected output
        return np.nan, np.nan, 0.0, gt_total_flight_duration_s

    common_min_time = max(mono_segment_aligned['Mono_Time_Synced_To_Opti'].min(), gt_segment_aligned['Time'].min())
    common_max_time = min(mono_segment_aligned['Mono_Time_Synced_To_Opti'].max(), gt_segment_aligned['Time'].max())

    common_duration_s = max(0, common_max_time - common_min_time)
    if common_duration_s < 1e-6 or len(mono_segment_aligned) < 2 or len(gt_segment_aligned) < 2:
        # Return four values
        return np.nan, np.nan, common_duration_s, gt_total_flight_duration_s

    query_times = mono_segment_aligned[(mono_segment_aligned['Mono_Time_Synced_To_Opti'] >= common_min_time) & (
            mono_segment_aligned['Mono_Time_Synced_To_Opti'] <= common_max_time)]['Mono_Time_Synced_To_Opti'].values
    if len(query_times) < 2:
        # Return four values
        return np.nan, np.nan, common_duration_s, gt_total_flight_duration_s

    mono_positions_common = mono_segment_aligned[mono_segment_aligned['Mono_Time_Synced_To_Opti'].isin(query_times)][
        ['x_wcs', 'y_wcs', 'z_wcs']].values
    try:
        interp_x_gt = interp1d(gt_segment_aligned['Time'], gt_segment_aligned['Ball_X'], kind='linear',
                               fill_value="extrapolate", bounds_error=False)
        interp_y_gt = interp1d(gt_segment_aligned['Time'], gt_segment_aligned['Ball_Y'], kind='linear',
                               fill_value="extrapolate", bounds_error=False)
        interp_z_gt = interp1d(gt_segment_aligned['Time'], gt_segment_aligned['Ball_Z'], kind='linear',
                               fill_value="extrapolate", bounds_error=False)
    except ValueError:
        # Return four values
        return np.nan, np.nan, common_duration_s, gt_total_flight_duration_s

    gt_interp_positions = np.vstack([interp_x_gt(query_times), interp_y_gt(query_times), interp_z_gt(query_times)]).T
    valid_indices = ~np.isnan(gt_interp_positions).any(axis=1) & ~np.isnan(mono_positions_common).any(axis=1)
    if not np.any(valid_indices):
        # Return four values
        return np.nan, np.nan, common_duration_s, gt_total_flight_duration_s

    mono_positions_common, gt_interp_positions = mono_positions_common[valid_indices], gt_interp_positions[
        valid_indices]
    if mono_positions_common.shape[0] < 2:
        # Return four values
        return np.nan, np.nan, common_duration_s, gt_total_flight_duration_s

    errors_3d = np.linalg.norm(mono_positions_common - gt_interp_positions, axis=1)
    rmse_3d, max_dev_3d = np.sqrt(np.mean(errors_3d ** 2)), np.max(errors_3d)

    # Return all four calculated values
    return rmse_3d, max_dev_3d, common_duration_s, gt_total_flight_duration_s


def refit_monocular_trajectory(mono_segment_aligned, gopro_fps):
    """Takes a T_GT-aligned monocular trajectory segment and re-fits it."""
    if mono_segment_aligned.empty or len(mono_segment_aligned) < 21:
        return None
    mono_wcs_data_for_fitting = [{'frame': row['frame'], 'Pc_wcs': row[['x_wcs', 'y_wcs', 'z_wcs']].values} for _, row
                                 in mono_segment_aligned.iterrows()]
    initial_p0_guess = mono_wcs_data_for_fitting[0]['Pc_wcs']
    initial_v_guess = [8.0, 45.0, 0.0]
    if len(mono_wcs_data_for_fitting) > 5:
        p_start, p_end = mono_wcs_data_for_fitting[0]['Pc_wcs'], mono_wcs_data_for_fitting[5]['Pc_wcs']
        t_start, t_end = mono_wcs_data_for_fitting[0]['frame'] / gopro_fps, mono_wcs_data_for_fitting[5][
            'frame'] / gopro_fps
        if (t_end - t_start) > 1e-6:
            v_vec = (p_end - p_start) / (t_end - t_start)
            speed, elev, azim = np.linalg.norm(v_vec), np.degrees(
                np.arctan2(v_vec[1], np.sqrt(v_vec[0] ** 2 + v_vec[2] ** 2))), np.degrees(
                np.arctan2(v_vec[2], v_vec[0]))
            initial_v_guess = [speed, elev, azim]
    return tu.fit_shot_trajectory_parameters(mono_wcs_data_for_fitting, gopro_fps, initial_p0_guess, initial_v_guess)


def plot_per_shot_overlay(shot_id, gt_traj, mono_traj_wcs, output_path):
    fig = go.Figure()
    if not gt_traj.empty:
        fig.add_trace(go.Scatter3d(x=gt_traj['Ball_X'], y=gt_traj['Ball_Z'], z=gt_traj['Ball_Y'], mode='lines+markers',
                                   name='Ground Truth (OptiTrack)', line=dict(color='green', width=4, dash='dash'),
                                   marker=dict(size=2)))
    if not mono_traj_wcs.empty:
        fig.add_trace(go.Scatter3d(x=mono_traj_wcs['x_wcs'], y=mono_traj_wcs['z_wcs'], z=mono_traj_wcs['y_wcs'],
                                   mode='markers+lines', name='Monocular System', line=dict(color='blue', width=4),
                                   marker=dict(size=2)))
    fig.update_layout(title=f'Shot {shot_id}: Trajectory Comparison (3D WCS)',
                      scene=dict(xaxis_title='X_wcs (m) [Towards Target]', yaxis_title='Z_wcs (m) [Lateral]',
                                 zaxis_title='Y_wcs (m) [Height]', aspectmode='data',
                                 camera_eye=dict(x=1.2, y=-2, z=0.9)), margin=dict(l=0, r=0, b=0, t=40))
    plot_dir = os.path.dirname(output_path)
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    fig.write_html(output_path)


def plot_master_trajectories(all_mono_trajectories, all_gt_trajectories, output_path):
    """Plots all trajectories on a single 3D plot with mean trajectories and std dev envelopes."""
    fig = go.Figure()

    def get_mean_std_trajectory(traj_list, x_col, y_col, z_col):
        """Calculates mean and std dev by interpolating against a common X-axis."""
        if not traj_list: return [None] * 5

        # Find the min start_x and use the global max_x
        min_x = min([t[x_col].min() for t in traj_list if not t.empty])
        max_x = MAX_X_WCS_FOR_ANALYSIS

        x_interp = np.linspace(min_x, max_x, 75)
        all_interp_y, all_interp_z = [], []

        for traj in traj_list:
            if traj.empty or len(traj) < 2: continue
            traj_unique_x = traj.drop_duplicates(subset=[x_col])
            if len(traj_unique_x) < 2: continue

            try:
                interp_y_of_x = interp1d(traj_unique_x[x_col], traj_unique_x[y_col], kind='linear', bounds_error=False,
                                         fill_value=np.nan)
                interp_z_of_x = interp1d(traj_unique_x[x_col], traj_unique_x[z_col], kind='linear', bounds_error=False,
                                         fill_value=np.nan)
                all_interp_y.append(interp_y_of_x(x_interp))
                all_interp_z.append(interp_z_of_x(x_interp))
            except ValueError:
                continue

        if not all_interp_y: return [None] * 5

        mean_y = np.nanmean(np.array(all_interp_y), axis=0)
        mean_z = np.nanmean(np.array(all_interp_z), axis=0)
        std_y = np.nanstd(np.array(all_interp_y), axis=0)
        std_z = np.nanstd(np.array(all_interp_z), axis=0)

        return x_interp, mean_y, mean_z, std_y, std_z

    def create_envelope_surface(mean_x, mean_y, mean_z, std_y, std_z, color, name, legendgroup):
        """Helper function to create a tube-like surface."""
        if any(v is None for v in [mean_x, mean_y, mean_z, std_y, std_z]): return None

        valid_idx = ~np.isnan(mean_x) & ~np.isnan(mean_y) & ~np.isnan(mean_z)
        mean_x, mean_y, mean_z = mean_x[valid_idx], mean_y[valid_idx], mean_z[valid_idx]
        std_y, std_z = std_y[valid_idx], std_z[valid_idx]
        if len(mean_x) < 2: return None

        N_angles = 20
        theta = np.linspace(0, 2 * np.pi, N_angles)

        x_surf = np.outer(mean_x, np.ones(N_angles))
        y_surf = np.outer(mean_z, np.ones(N_angles)) + np.outer(std_z, np.sin(theta))
        z_surf = np.outer(mean_y, np.ones(N_angles)) + np.outer(std_y, np.cos(theta))

        return go.Surface(x=x_surf, y=y_surf, z=z_surf,
                          colorscale=[[0, color], [1, color]],
                          opacity=1, showscale=False,
                          name=f'{name} Spread (±1 std dev)', legendgroup=legendgroup)

    # --- Truncate trajectories before processing for master plot ---
    truncated_gt_trajectories = [truncate_segment_at_x_wcs(traj, 'Ball_X', MAX_X_WCS_FOR_ANALYSIS) for traj in
                                 all_gt_trajectories]
    truncated_mono_trajectories = [truncate_segment_at_x_wcs(traj, 'x_wcs', MAX_X_WCS_FOR_ANALYSIS) for traj in
                                   all_mono_trajectories]

    # --- Process and Plot GT Data ---
    mean_x_gt, mean_y_gt, mean_z_gt, std_y_gt, std_z_gt = get_mean_std_trajectory(truncated_gt_trajectories, 'Ball_X',
                                                                                  'Ball_Y', 'Ball_Z')
    if mean_x_gt is not None:
        gt_surface = create_envelope_surface(mean_x_gt, mean_y_gt, mean_z_gt, std_y_gt, std_z_gt, 'lightgreen', 'GT',
                                             'gt')
        if gt_surface: fig.add_trace(gt_surface)
        fig.add_trace(go.Scatter3d(x=mean_x_gt, y=mean_z_gt, z=mean_y_gt, mode='lines', name='Mean GT Trajectory',
                                   line=dict(color='darkgreen', width=0), legendgroup='gt'))

    # --- Process and Plot Monocular Data ---
    mean_x_mono, mean_y_mono, mean_z_mono, std_y_mono, std_z_mono = get_mean_std_trajectory(truncated_mono_trajectories,
                                                                                            'x_wcs', 'y_wcs', 'z_wcs')
    if mean_x_mono is not None:
        mono_surface = create_envelope_surface(mean_x_mono, mean_y_mono, mean_z_mono, std_y_mono, std_z_mono,
                                               'lightblue', 'Mono', 'mono')
        if mono_surface: fig.add_trace(mono_surface)
        fig.add_trace(
            go.Scatter3d(x=mean_x_mono, y=mean_z_mono, z=mean_y_mono, mode='lines', name='Mean Monocular Trajectory',
                         line=dict(color='darkblue', width=0), legendgroup='mono'))

    fig.update_layout(
        title='Master Plot of Mean Trajectories and Standard Deviation Envelopes (N=67)',
        scene=dict(
            xaxis_title='X_wcs (m) [Towards Target]',
            yaxis_title='Z_wcs (m) [Lateral]',
            zaxis_title='Y_wcs (m) [Height]',
            aspectmode='data',
            camera_eye=dict(x=1.5, y=-2.5, z=1.0)
        ),
        legend_title_text='Trajectory Type',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.write_html(output_path)
    print(f"Master trajectory plot saved to {output_path}")

def main():
    all_batches = sorted([d for d in os.listdir(EXPERIMENT_DATA_ROOT) if
                          d.startswith("batch") and os.path.isdir(os.path.join(EXPERIMENT_DATA_ROOT, d))])
    if not all_batches: print(f"No 'batchX' folders found in {EXPERIMENT_DATA_ROOT}."); return

    master_results_list = []
    all_mono_trajectories_for_master_plot = []
    all_gt_trajectories_for_master_plot = []

    for session_name in all_batches:
        print(f"\n================ Processing Session: {session_name} ================")
        session_dir = os.path.join(EXPERIMENT_DATA_ROOT, session_name)
        analysis_output_dir = os.path.join(session_dir, "analysis_results_final")
        os.makedirs(os.path.join(analysis_output_dir, "per_shot_plots"), exist_ok=True)
        K_cam, D_cam, R_wcs_to_ccs, tvec_wcs_to_ccs = load_batch_extrinsics(session_dir)

        shot_folders = sorted([d for d in os.listdir(session_dir) if
                               d.startswith("shot_") and os.path.isdir(os.path.join(session_dir, d))])
        if not shot_folders: continue

        for shot_folder_name in shot_folders:
            shot_id = int(shot_folder_name.split('_')[1])
            if session_name in EXCLUSION_LIST and shot_id in EXCLUSION_LIST.get(session_name, []):
                print(f"\n--- Skipping {shot_folder_name} (in exclusion list) ---")
                continue

            print(f"\n--- Analyzing {shot_folder_name} ---")
            shot_dir = os.path.join(session_dir, shot_folder_name)
            gt_summary, mono_summary_orig, gt_traj, mono_traj_wcs = load_shot_data(shot_dir)
            if gt_summary is None: continue

            all_mono_trajectories_for_master_plot.append(mono_traj_wcs)
            all_gt_trajectories_for_master_plot.append(gt_traj)

            # --- 1. Original End-to-End Metrics ---
            temporal_error_s = abs(
                mono_summary_orig.get('GoPro_Frame_Detected', 0) - gt_summary.get('GoPro_Frame_Equivalent',
                                                                                  0)) / GOPRO_FPS
            p_mono_orig = np.array(mono_summary_orig.get('P0_wcs_fitted', [np.nan] * 3))
            p_gt = np.array(gt_summary.get('P0_wcs_fitted', [np.nan] * 3))
            spatial_error_m_orig = np.linalg.norm(p_mono_orig - p_gt) if not np.isnan(p_mono_orig).any() else np.nan
            speed_error_orig = abs(
                mono_summary_orig.get('Speed_mps_wcs_fitted', np.nan) - gt_summary.get('Speed_mps_wcs_fitted', np.nan))
            elev_error_orig = abs(
                mono_summary_orig.get('Angle_Elev_wcs_fitted', np.nan) - gt_summary.get('Angle_Elev_wcs_fitted',
                                                                                        np.nan))
            azim_error_orig = abs(
                mono_summary_orig.get('Angle_Azim_wcs_fitted', np.nan) - gt_summary.get('Angle_Azim_wcs_fitted',
                                                                                        np.nan))

            # --- 2. Aligned Analysis (Supervisor's Method) ---
            gt_release_time = gt_summary.get('T_OptiTrackRelease', -1)
            if gt_release_time == -1: continue
            time_offset = gt_traj['Time'].iloc[0] - (gt_traj['GoPro_Frame_Synced'].iloc[
                                                         0] / GOPRO_FPS) if 'GoPro_Frame_Synced' in gt_traj.columns else 0

            gt_segment_aligned = gt_traj[gt_traj['Time'] >= gt_release_time]
            mono_traj_wcs['Mono_Time_Synced_To_Opti'] = (mono_traj_wcs['frame'] / GOPRO_FPS) + time_offset
            mono_segment_aligned = mono_traj_wcs[mono_traj_wcs['Mono_Time_Synced_To_Opti'] >= gt_release_time]

            # --- Calculate Initial Spatial Offset (new definition) ---
            initial_spatial_offset_m = np.nan
            p_gt_for_offset = np.array(gt_summary.get('P0_wcs_fitted', [np.nan] * 3))

            if not mono_traj_wcs.empty and not np.isnan(gt_release_time) and gt_release_time != -1 and not np.isnan(
                    p_gt_for_offset).any():
                mono_times_aligned = mono_traj_wcs['Mono_Time_Synced_To_Opti']
                if not mono_times_aligned.empty and \
                        (gt_release_time >= mono_times_aligned.min()) and \
                        (gt_release_time <= mono_times_aligned.max()):
                    try:
                        interp_x_mono = interp1d(mono_times_aligned, mono_traj_wcs['x_wcs'], kind='linear',
                                                 fill_value="extrapolate", bounds_error=False)
                        interp_y_mono = interp1d(mono_times_aligned, mono_traj_wcs['y_wcs'], kind='linear',
                                                 fill_value="extrapolate", bounds_error=False)
                        interp_z_mono = interp1d(mono_times_aligned, mono_traj_wcs['z_wcs'], kind='linear',
                                                 fill_value="extrapolate", bounds_error=False)
                        p_mono_at_gt_release = np.array([
                            interp_x_mono(gt_release_time).item(),
                            interp_y_mono(gt_release_time).item(),
                            interp_z_mono(gt_release_time).item()
                        ])
                        if not np.isnan(p_mono_at_gt_release).any():
                            initial_spatial_offset_m = np.linalg.norm(p_mono_at_gt_release - p_gt_for_offset)
                    except ValueError:
                        initial_spatial_offset_m = np.nan
                else:
                    initial_spatial_offset_m = np.nan

            gt_segment_aligned_trunc = truncate_segment_at_x_wcs(gt_segment_aligned, 'Ball_X', MAX_X_WCS_FOR_ANALYSIS)
            mono_segment_aligned_trunc = truncate_segment_at_x_wcs(mono_segment_aligned, 'x_wcs',
                                                                   MAX_X_WCS_FOR_ANALYSIS)

            mono_summary_refit = refit_monocular_trajectory(mono_segment_aligned_trunc, GOPRO_FPS)

            speed_error_refit, elev_error_refit, azim_error_refit = np.nan, np.nan, np.nan
            if mono_summary_refit and mono_summary_refit.get('success'):
                speed_error_refit = abs(
                    mono_summary_refit['speed_mps_wcs'] - gt_summary.get('Speed_mps_wcs_fitted', np.nan))
                elev_error_refit = abs(
                    mono_summary_refit['angle_deg_elev_wcs'] - gt_summary.get('Angle_Elev_wcs_fitted', np.nan))
                azim_error_refit = abs(
                    mono_summary_refit['angle_deg_azim_wcs'] - gt_summary.get('Angle_Azim_wcs_fitted', np.nan))

            # --- MODIFIED: Capture the max_dev value ---
            rmse_3d_aligned, max_dev_3d_aligned, common_dur, gt_flight_dur = calculate_aligned_rmse(
                gt_segment_aligned_trunc, mono_segment_aligned_trunc)
            percentage_tracked = (common_dur / gt_flight_dur * 100) if gt_flight_dur > 1e-6 else 0.0

            # --- ADDED: Include the new metric in the results dictionary ---
            master_results_list.append({
                'batch': session_name, 'shot_id': shot_id,
                'temporal_error_s': temporal_error_s,
                'spatial_error_m_orig': spatial_error_m_orig,
                'initial_spatial_offset_m': initial_spatial_offset_m,
                'speed_error_mps_orig': speed_error_orig,
                'elevation_error_deg_orig': elev_error_orig,
                'azimuth_error_deg_orig': azim_error_orig,
                'trajectory_rmse_3d_aligned_m': rmse_3d_aligned,
                'max_deviation_3d_aligned_m': max_dev_3d_aligned,  # <-- NEW METRIC HERE
                'percentage_trajectory_tracked': percentage_tracked,
                'speed_error_mps_refit': speed_error_refit,
                'elevation_error_deg_refit': elev_error_refit,
                'azimuth_error_deg_refit': azim_error_refit,
                'gt_speed': gt_summary.get('Speed_mps_wcs_fitted', np.nan),
                'mono_speed_orig': mono_summary_orig.get('Speed_mps_wcs_fitted', np.nan),
                'mono_speed_refit': mono_summary_refit.get('speed_mps_wcs', np.nan) if mono_summary_refit else np.nan,
                'gt_elevation': gt_summary.get('Angle_Elev_wcs_fitted', np.nan),
                'mono_elevation_orig': mono_summary_orig.get('Angle_Elev_wcs_fitted', np.nan),
                'mono_elevation_refit': mono_summary_refit.get('angle_deg_elev_wcs',
                                                               np.nan) if mono_summary_refit else np.nan,
                'gt_azimuth': gt_summary.get('Angle_Azim_wcs_fitted', np.nan),
                'mono_azimuth_orig': mono_summary_orig.get('Angle_Azim_wcs_fitted', np.nan),
                'mono_azimuth_refit': mono_summary_refit.get('angle_deg_azim_wcs',
                                                             np.nan) if mono_summary_refit else np.nan,
            })

    if not master_results_list: print("No shots analyzed."); return
    results_df = pd.DataFrame(master_results_list)
    final_output_dir = os.path.join(EXPERIMENT_DATA_ROOT, "master_analysis_results_final")
    os.makedirs(final_output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(final_output_dir, "all_batches_comparison_final.csv"), index=False)

    if all_mono_trajectories_for_master_plot and all_gt_trajectories_for_master_plot:
        master_plot_path = os.path.join(final_output_dir, "master_trajectory_overlay.html")
        plot_master_trajectories(all_mono_trajectories_for_master_plot, all_gt_trajectories_for_master_plot,
                                 master_plot_path)

    summary_stats = results_df.drop(columns=['batch', 'shot_id']).describe().transpose()
    summary_stats['count_valid'] = results_df.drop(columns=['batch', 'shot_id']).count()
    summary_stats['IQR'] = summary_stats['75%'] - summary_stats['25%']
    print("\n--- Overall Performance Metrics (All Batches Combined) ---")
    print(summary_stats[['count_valid', 'mean', 'std', '50%', 'min', 'max', 'IQR']].rename(columns={'50%': 'median'}))
    summary_stats.to_csv(os.path.join(final_output_dir, "summary_statistics_final.csv"))

    # --- ADDED: Include the new metric in the list of plots to generate ---
    error_cols_to_plot = [
        'temporal_error_s', 'spatial_error_m_orig', 'initial_spatial_offset_m',
        'trajectory_rmse_3d_aligned_m',
        'max_deviation_3d_aligned_m',  # <-- NEW METRIC HERE
        'speed_error_mps_orig', 'elevation_error_deg_orig', 'azimuth_error_deg_orig',
        'speed_error_mps_refit', 'elevation_error_deg_refit', 'azimuth_error_deg_refit'
    ]
    for col in error_cols_to_plot:
        if col not in results_df.columns or results_df[col].notna().sum() < 2: continue
        plt.figure(figsize=(8, 5))
        results_df[col].dropna().hist(bins=15)
        plt.title(f'Distribution of {col.replace("_", " ").title()} (All Batches)')
        unit = "(s)" if "_s" in col else "(m)" if "_m" in col else "(m/s)" if "mps" in col else "(degrees)" if "deg" in col else ""
        plt.xlabel(f'{col.replace("_", " ").title()} {unit}')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(final_output_dir, f"hist_{col}.png"))
        plt.close()

    for param in ['speed', 'elevation', 'azimuth']:
        for fit_type in ['orig', 'refit']:
            gt_col = f'gt_{param}'
            mono_col = f'mono_{param}_{fit_type}'
            if gt_col not in results_df.columns or mono_col not in results_df.columns: continue

            valid_data = results_df[[gt_col, mono_col]].dropna()
            if len(valid_data) < 2: continue

            diffs = valid_data[mono_col] - valid_data[gt_col]
            means = (valid_data[mono_col] + valid_data[gt_col]) / 2
            mean_diff, std_diff = np.mean(diffs), np.std(diffs)
            lower_loa, upper_loa = mean_diff - 1.96 * std_diff, mean_diff + 1.96 * std_diff

            plt.figure(figsize=(9, 6))
            plt.scatter(means, diffs, alpha=0.6, label=f'Individual Shots (N={len(valid_data)})', s=15)
            plt.axhline(mean_diff, color='red', linestyle='--', label=f'Mean Difference: {mean_diff:.2f}')
            plt.axhline(upper_loa, color='dimgray', linestyle=':', label=f'95% LoA: {upper_loa:.2f}')
            plt.axhline(lower_loa, color='dimgray', linestyle=':', label=f'_nolegend_')
            plt.fill_between(plt.xlim(), lower_loa, upper_loa, color='lightgray', alpha=0.3,
                             label='95% Limits of Agreement')

            unit = "(m/s)" if param == "speed" else "(degrees)"
            plt.title(f'Bland-Altman Plot for {param.title()} ({fit_type.title()} Fit)', fontsize=14)
            plt.xlabel(f'Mean of Measurements ({param.title()} {unit})', fontsize=12)
            plt.ylabel(f'Difference (Monocular - GT) {unit}', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.7);
            plt.tight_layout()
            plt.savefig(os.path.join(final_output_dir, f"bland_altman_{param}_{fit_type}.png"))
            plt.close()

    if 'percentage_trajectory_tracked' in results_df.columns and results_df[
        'percentage_trajectory_tracked'].notna().sum() > 1:
        plt.figure(figsize=(8, 5))
        results_df['percentage_trajectory_tracked'].dropna().hist(bins=np.arange(0, 101, 10))
        plt.title('Distribution of Percentage of Relevant GT Trajectory Tracked')
        plt.xlabel('Percentage of GT Trajectory Tracked (%)')
        plt.ylabel('Frequency (Number of Shots)')
        plt.xticks(np.arange(0, 101, 10))
        plt.grid(axis='y', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(final_output_dir, "hist_percentage_trajectory_tracked.png"))
        plt.close()

    print(f"\nAnalysis complete. All aggregate results saved in:\n{final_output_dir}")


if __name__ == '__main__':
    main()