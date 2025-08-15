
import open3d as o3d
import numpy as np
import copy

from icp_algorithm import (
    load_point_clouds,
    preprocess_point_cloud,
    compute_se3_pose_error,
    generate_perturbed_initial_guess_from_pose,
    compute_pose
)

from icp_scratch import (
    compare_correspondence_methods,
    icp_point_to_plane_with_method_choice
)

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"


def visualize_results(pcd_source, pcd_target, T_initial_guess, T_o3d_p2l, T_basic, T_normals):
    # Prepare visualization geometries
    pcd_target_viz = copy.deepcopy(pcd_target)
    pcd_target_viz.paint_uniform_color([0, 0, 1])  # Blue for target
    
    # Initial guess
    pcd_initial = copy.deepcopy(pcd_source)
    pcd_initial.transform(T_initial_guess)
    pcd_initial.paint_uniform_color([1, 1, 0])     # Yellow for initial

    # Open3D method result
    pcd_o3d = copy.deepcopy(pcd_source)
    pcd_o3d.transform(T_o3d_p2l)
    pcd_o3d.paint_uniform_color([1, 0, 1])         # Magenta for Open3D

    # Basic method result
    pcd_basic = copy.deepcopy(pcd_source)
    pcd_basic.transform(T_basic)
    pcd_basic.paint_uniform_color([0, 1, 0])       # Green for basic method

    # Normal-aware method result
    pcd_normals = copy.deepcopy(pcd_source)
    pcd_normals.transform(T_normals)
    pcd_normals.paint_uniform_color([1, 0, 0])     # Red for normal-aware method

    # Show initial alignment
    print("0. Initial guess alignment (Yellow + Blue)")
    o3d.visualization.draw_geometries([pcd_initial, pcd_target_viz],
                                      window_name="Initial Alignment")

    # Show Open3D alignment
    print("1. Open3D alignment (Magenta + Blue)")
    o3d.visualization.draw_geometries([pcd_o3d, pcd_target_viz],
                                      window_name="Initial Alignment")

    # Show basic method result
    print("2. Basic Euclidean ICP result (Red + Blue)")
    o3d.visualization.draw_geometries([pcd_basic, pcd_target_viz],
                                      window_name="Basic Euclidean ICP")
    
    # Show normal-aware method result
    print("3. Normal-aware ICP result (Green + Blue)")
    o3d.visualization.draw_geometries([pcd_normals, pcd_target_viz],
                                      window_name="Normal-Aware ICP")
    
def normal_space_sampling(pcd, n_samples=1000, n_theta=6, n_phi=12, 
                         ground_fraction_cap=0.15, vertical_boost=2.0):
    """
    Normal-space sampling following the practical recipe.
    
    This addresses question (1): "how to sample queries so ground doesn't dominate"
    
    Args:
        pcd: Point cloud with normals
        n_samples: Target number of samples
        n_theta: Number of inclination bins [0, π]
        n_phi: Number of azimuth bins [-π, π]
        ground_fraction_cap: Maximum fraction of samples from ground-like normals
        vertical_boost: Multiplier for vertical-ish features (|n_z| < 0.3)
        
    Returns:
        o3d.geometry.PointCloud: Sampled point cloud with diverse normals
    """
    if not pcd.has_normals():
        raise ValueError("Point cloud must have normals for geometry-aware sampling")
    
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    n_points = len(points)
    
    if n_points <= n_samples:
        return pcd
    
    # Make normals consistently oriented (flip so n·z >= 0)
    normals = np.copy(normals)
    flip_mask = normals[:, 2] < 0
    normals[flip_mask] *= -1
    
    # Convert to spherical coordinates following the recipe
    # theta = arccos(n_z) ∈ [0, π] (inclination)
    # phi = atan2(n_y, n_x) ∈ (-π, π] (azimuth)
    theta = np.arccos(np.clip(normals[:, 2], -1, 1))  # [0, π]
    phi = np.arctan2(normals[:, 1], normals[:, 0])    # [-π, π]
    
    # Quantize into bins
    theta_bins = np.linspace(0, np.pi, n_theta + 1)
    phi_bins = np.linspace(-np.pi, np.pi, n_phi + 1)
    
    theta_indices = np.digitize(theta, theta_bins) - 1
    phi_indices = np.digitize(phi, phi_bins) - 1
    
    # Ensure indices are within bounds
    theta_indices = np.clip(theta_indices, 0, n_theta - 1)
    phi_indices = np.clip(phi_indices, 0, n_phi - 1)
    
    # Create bin labels
    bin_labels = theta_indices * n_phi + phi_indices
    unique_bins = np.unique(bin_labels)
    
    # Identify ground-like bins (high theta values, close to π/2)
    # Ground normals have theta close to 0 (pointing up)
    ground_threshold = np.pi / 6  # 30 degrees from vertical
    
    # Assign quotas with bias
    total_bins = len(unique_bins)
    selected_indices = []
    quota_used = 0
    
    # Calculate base quota per bin
    base_quota = n_samples // total_bins
    
    for bin_label in unique_bins:
        bin_mask = bin_labels == bin_label
        bin_indices = np.where(bin_mask)[0]
        
        if len(bin_indices) == 0:
            continue
            
        # Determine bin characteristics
        bin_theta = theta_indices[bin_indices[0]]
        bin_theta_value = (bin_theta + 0.5) * np.pi / n_theta
        
        # Check if this is a ground bin (small theta, pointing up)
        is_ground = bin_theta_value < ground_threshold
        
        # Check if this is a vertical bin (high |n_z| < 0.3 means more horizontal)
        sample_normal = normals[bin_indices[0]]
        is_vertical = abs(sample_normal[2]) < 0.3
        
        # Calculate quota for this bin
        quota = base_quota
        if is_vertical:
            quota = int(quota * vertical_boost)
        
        # Cap ground bins
        if is_ground:
            max_ground_quota = int(n_samples * ground_fraction_cap / 10)  # Distributed among ground bins
            quota = min(quota, max_ground_quota)
        
        # Don't exceed remaining samples
        remaining_samples = n_samples - quota_used
        quota = min(quota, remaining_samples)
        quota = min(quota, len(bin_indices))
        
        if quota <= 0:
            continue
        
        # Sample within this bin (prefer voxel-grid approach for spatial spread)
        if len(bin_indices) <= quota:
            selected_indices.extend(bin_indices)
        else:
            # Create a mini point cloud for this bin and voxel downsample
            bin_points = points[bin_indices]
            bin_pcd = o3d.geometry.PointCloud()
            bin_pcd.points = o3d.utility.Vector3dVector(bin_points)
            
            # Voxel downsample within the bin for spatial spread
            voxel_size = 0.3  # Adaptive based on point density
            if len(bin_indices) > quota * 3:
                downsampled = bin_pcd.voxel_down_sample(voxel_size)
                downsampled_indices = []
                for dp in np.asarray(downsampled.points):
                    # Find closest original point
                    distances = np.linalg.norm(bin_points - dp, axis=1)
                    closest_idx = np.argmin(distances)
                    downsampled_indices.append(bin_indices[closest_idx])
                
                # If we still have too many, randomly subsample
                if len(downsampled_indices) > quota:
                    downsampled_indices = np.random.choice(downsampled_indices, quota, replace=False)
                selected_indices.extend(downsampled_indices)
            else:
                # Random sampling if voxel downsampling isn't effective
                sampled = np.random.choice(bin_indices, quota, replace=False)
                selected_indices.extend(sampled)
        
        quota_used += quota
        if quota_used >= n_samples:
            break
    
    # Create new point cloud with selected points
    selected_indices = np.array(selected_indices[:n_samples])  # Ensure we don't exceed
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(points[selected_indices])
    sampled_pcd.normals = o3d.utility.Vector3dVector(normals[selected_indices])
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        sampled_pcd.colors = o3d.utility.Vector3dVector(colors[selected_indices])
    
    print(f"Normal-space sampling: {n_points} -> {len(selected_indices)} points")
    print(f"Covered {len(unique_bins)} normal direction bins ({n_theta}×{n_phi})")
    
    return sampled_pcd


def main():

    np.random.seed(42)
    # Load and preprocess point clouds
    pcd_source, pcd_target = load_point_clouds()

    # preprocess pointclouds
    voxel_size = 0.1
    # pcd_source_processed = preprocess_point_cloud(pcd_source, voxel_size)
    pcd_target_processed = preprocess_point_cloud(pcd_target, voxel_size)



    ####### Demonstrate normal-space sampling
    print(f"\nDemonstrating normal-space sampling...")
    pcd_source_pre = preprocess_point_cloud(pcd_source, voxel_size)
    n_samples = min(1000, int(len(pcd_source_pre.points) * 0.8))
    pcd_source_processed = normal_space_sampling(pcd_source_pre, n_samples, n_theta=6, n_phi=12)



    # convert to numpy arrays
    source_points = np.asarray(pcd_source_processed.points)
    target_points = np.asarray(pcd_target_processed.points)
    source_normals = np.asarray(pcd_source_processed.normals)
    target_normals = np.asarray(pcd_target_processed.normals)

    # setup ground truth and initial guess
    lidar_pose1 = {'x': 0.0, 'y': 0.0, 'z': 1.5, 
                   'roll_deg': 0.0, 'pitch_deg': 0.0, 'yaw_deg': 0.0}
    lidar_pose2 = {'x': 0.2, 'y': 0.5, 'z': 1.5, 
                   'roll_deg': 0.0, 'pitch_deg': 0.0, 'yaw_deg': 20.0}
    
    T_lidar_source = compute_pose(**lidar_pose1)
    T_lidar_target = compute_pose(**lidar_pose2)
    T_ground_truth = np.linalg.inv(T_lidar_target) @ T_lidar_source

    # Generate perturbed initial guess
    perturb_values = {'dx': 0.0, 'dy': 0.0, 'dz': 0.0, 
                      'droll_deg': 0.0, 'dpitch_deg': 0.0, 'dyaw_deg': -8.0}
    T_initial_guess = generate_perturbed_initial_guess_from_pose(T_ground_truth, **perturb_values)

    print(f"\nGround truth transformation:")
    print(T_ground_truth)
    print(f"\nInitial guess transformation:")
    print(T_initial_guess)


    # Apply initial transformation to source
    # T @ [p; 1] for homogeneous coordinates
    points_homogeneous = np.hstack([source_points, np.ones((len(source_points), 1))])
    transformed_homogeneous = (T_initial_guess @ points_homogeneous.T).T
    source_initial = transformed_homogeneous[:, :3]
    
    # For normals, only apply rotation (no translation)
    R_initial = T_initial_guess[:3, :3]
    source_normals_initial = (R_initial @ source_normals.T).T
    
    # Compare correspondence methods
    compare_correspondence_methods(
        source_initial, target_points, source_normals_initial, target_normals,
        max_distance=5.0, max_normal_angle=60.0, verbose=True
    )

    # compare ICP with different methods
    # Method 0: Open3D ICP
    print(BLUE + f"\n--- Open3d Point-to-Plane ICP ---" + RESET)
    open3d_max_correspondence_distance = 50
    result_o3d_p2l = o3d.pipelines.registration.registration_icp(
        pcd_source_processed, pcd_target_processed, open3d_max_correspondence_distance, T_initial_guess,
        # o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    print(GREEN + f"\n--- Open3d P2L ICP Done ---" + RESET)
    # NOTE: increase the open3d_threshold to 50.0, 
    # the point-to-plane ICP works but the point-to-point ICP still fails


    # Method 1: Basic Euclidean correspondences
    print(BLUE + f"\n--- ICP with Basic Euclidean Correspondences ---" + RESET)
    result_basic = icp_point_to_plane_with_method_choice(
        source_points, target_points, source_normals, target_normals,
        initial_guess=T_initial_guess,
        max_iterations=100,
        tolerance=1e-6,
        max_correspondence_distance=50.0,
        use_normals=False,  # Force basic method
        verbose=True
    )
    print(GREEN + f"\n--- ICP with Basic Euclidean Correspondences Done ---" + RESET)

    # Method 2: Normal-aware correspondences
    print(BLUE + f"\n--- ICP with Normal-Aware Correspondences ---" + RESET)
    result_normals = icp_point_to_plane_with_method_choice(
        source_points, target_points, source_normals, target_normals,
        initial_guess=T_initial_guess,
        max_iterations=100,
        tolerance=1e-6,
        max_correspondence_distance=50.0,
        use_normals=True,   # Use normal-aware method
        verbose=True
    )
    print(GREEN + f"\n--- ICP with Normal-Aware Correspondences Done ---" + RESET)

    visualize_results(pcd_source_processed, 
                      pcd_target_processed,
                      T_initial_guess,
                      result_o3d_p2l.transformation,
                      result_basic.transformation, 
                      result_normals.transformation)

if __name__ == "__main__":
    main()
