#!/usr/bin/env python3
"""
script to test the data association for point cloud registration.

This script demonstrates the difference between basic Euclidean correspondence 
and normal-aware correspondence in the custom ICP implementation.
"""

import open3d as o3d
import numpy as np
import copy

# Import functions from existing ICP algorithm
from icp_algorithm import (
    load_point_clouds, 
    preprocess_point_cloud, 
    compute_se3_pose_error,
    generate_perturbed_initial_guess_from_pose,
    compute_pose
)

# Import our custom ICP implementations
from icp_scratch import (
    compare_correspondence_methods,
    icp_point_to_plane_with_method_choice,
    ICPResult
)

def main():
    print("="*80)
    print("COMPARING ICP CORRESPONDENCE METHODS")
    print("="*80)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Load and preprocess point clouds
    pcd_source, pcd_target = load_point_clouds()
    if pcd_source is None or pcd_target is None:
        print("Failed to load point clouds.")
        return
    
    # Preprocess point clouds with normals
    voxel_size = 0.1
    pcd_source_processed = preprocess_point_cloud(pcd_source, voxel_size)
    pcd_target_processed = preprocess_point_cloud(pcd_target, voxel_size)
    
    # Convert to numpy arrays
    source_points = np.asarray(pcd_source_processed.points)
    target_points = np.asarray(pcd_target_processed.points)
    source_normals = np.asarray(pcd_source_processed.normals)
    target_normals = np.asarray(pcd_target_processed.normals)
    
    print(f"Point clouds loaded:")
    print(f"  Source: {len(source_points)} points")
    print(f"  Target: {len(target_points)} points")
    print(f"  Normals available: {pcd_source_processed.has_normals() and pcd_target_processed.has_normals()}")
    
    # Set up ground truth and initial guess
    lidar_pose1 = {'x': 0.0, 'y': 0.0, 'z': 1.5, 'roll_deg': 0.0, 'pitch_deg': 0.0, 'yaw_deg': 0.0}
    lidar_pose2 = {'x': 0.2, 'y': 0.5, 'z': 1.5, 'roll_deg': 0.0, 'pitch_deg': 0.0, 'yaw_deg': 20.0}
    
    T_lidar_source = compute_pose(**lidar_pose1)
    T_lidar_target = compute_pose(**lidar_pose2)
    T_ground_truth = np.linalg.inv(T_lidar_target) @ T_lidar_source
    
    # Generate perturbed initial guess
    perturbation_values = {'dx': 0.5, 'dy': 0.2, 'dz': 0.0, 'droll_deg': 0.0, 'dpitch_deg': 0.0, 'dyaw_deg': 20.0}
    T_initial_guess = generate_perturbed_initial_guess_from_pose(T_ground_truth, **perturbation_values)
    
    print(f"\nGround truth transformation:")
    print(T_ground_truth)
    print(f"\nInitial guess transformation:")
    print(T_initial_guess)
    
    # Compare correspondence methods before running ICP
    print(f"\n" + "="*80)
    print("STEP 1: CORRESPONDENCE METHOD COMPARISON")
    print("="*80)
    
    # Apply initial transformation to source
    source_initial = source_points @ T_initial_guess[:3, :3].T + T_initial_guess[:3, 3]
    source_normals_initial = source_normals @ T_initial_guess[:3, :3].T
    
    # Compare correspondence methods
    corr_results = compare_correspondence_methods(
        source_initial, target_points, source_normals_initial, target_normals,
        max_distance=5.0, max_normal_angle=60.0, verbose=True
    )
    
    # Run ICP with both methods
    print(f"\n" + "="*80)
    print("STEP 2: ICP WITH DIFFERENT CORRESPONDENCE METHODS")
    print("="*80)
    
    # Method 1: Basic Euclidean correspondences
    print(f"\n--- ICP with Basic Euclidean Correspondences ---")
    result_basic = icp_point_to_plane_with_method_choice(
        source_points, target_points, source_normals, target_normals,
        initial_guess=T_initial_guess,
        max_iterations=100,
        tolerance=1e-6,
        max_correspondence_distance=50.0,
        use_normals=False,  # Force basic method
        verbose=True
    )
    
    # Method 2: Normal-aware correspondences
    print(f"\n--- ICP with Normal-Aware Correspondences ---")
    result_normals = icp_point_to_plane_with_method_choice(
        source_points, target_points, source_normals, target_normals,
        initial_guess=T_initial_guess,
        max_iterations=100,
        tolerance=1e-6,
        max_correspondence_distance=50.0,
        use_normals=True,   # Use normal-aware method
        verbose=True
    )
    
    # Compare results
    print(f"\n" + "="*80)
    print("STEP 3: RESULTS COMPARISON")
    print("="*80)
    
    # Calculate errors
    errors_basic = compute_se3_pose_error(result_basic.transformation, T_ground_truth)
    errors_normals = compute_se3_pose_error(result_normals.transformation, T_ground_truth)
    
    print(f"{'Metric':<30} {'Basic Euclidean':<20} {'Normal-Aware':<20} {'Improvement':<15}")
    print("-" * 85)
    print(f"{'Fitness':<30} {result_basic.fitness:<20.6f} {result_normals.fitness:<20.6f} {((result_normals.fitness - result_basic.fitness) / result_basic.fitness * 100 if result_basic.fitness > 0 else 0):<15.2f}%")
    print(f"{'Inlier RMSE (m)':<30} {result_basic.inlier_rmse:<20.6f} {result_normals.inlier_rmse:<20.6f} {((result_basic.inlier_rmse - result_normals.inlier_rmse) / result_basic.inlier_rmse * 100 if result_basic.inlier_rmse > 0 else 0):<15.2f}%")
    print(f"{'Translation Error (m)':<30} {errors_basic['translation']['error_norm']:<20.6f} {errors_normals['translation']['error_norm']:<20.6f} {((errors_basic['translation']['error_norm'] - errors_normals['translation']['error_norm']) / errors_basic['translation']['error_norm'] * 100 if errors_basic['translation']['error_norm'] > 0 else 0):<15.2f}%")
    print(f"{'Rotation Error (deg)':<30} {errors_basic['rotation']['error_norm_deg']:<20.6f} {errors_normals['rotation']['error_norm_deg']:<20.6f} {((errors_basic['rotation']['error_norm_deg'] - errors_normals['rotation']['error_norm_deg']) / errors_basic['rotation']['error_norm_deg'] * 100 if errors_basic['rotation']['error_norm_deg'] > 0 else 0):<15.2f}%")
    print(f"{'Iterations':<30} {result_basic.num_iterations:<20d} {result_normals.num_iterations:<20d} {result_normals.num_iterations - result_basic.num_iterations:<15d}")
    
    # Visualization
    print(f"\n" + "="*80)
    print("STEP 4: VISUALIZATION")
    print("="*80)
    
    # Prepare visualization geometries
    pcd_target_viz = copy.deepcopy(pcd_target_processed)
    pcd_target_viz.paint_uniform_color([0, 0, 1])  # Blue for target
    
    # Initial guess
    pcd_initial = copy.deepcopy(pcd_source_processed)
    pcd_initial.transform(T_initial_guess)
    pcd_initial.paint_uniform_color([1, 1, 0])  # Yellow for initial
    
    # Basic method result
    pcd_basic = copy.deepcopy(pcd_source_processed)
    pcd_basic.transform(result_basic.transformation)
    pcd_basic.paint_uniform_color([1, 0, 0])  # Red for basic method
    
    # Normal-aware method result
    pcd_normals = copy.deepcopy(pcd_source_processed)
    pcd_normals.transform(result_normals.transformation)
    pcd_normals.paint_uniform_color([0, 1, 0])  # Green for normal-aware method
    
    print("Displaying visualizations...")
    print("Colors: Yellow=Initial, Red=Basic, Green=Normal-aware, Blue=Target")
    
    # Show initial alignment
    print("1. Initial alignment (Yellow + Blue)")
    o3d.visualization.draw_geometries([pcd_initial, pcd_target_viz],
                                      window_name="Initial Alignment")
    
    # Show basic method result
    print("2. Basic Euclidean ICP result (Red + Blue)")
    o3d.visualization.draw_geometries([pcd_basic, pcd_target_viz],
                                      window_name="Basic Euclidean ICP")
    
    # Show normal-aware method result
    print("3. Normal-aware ICP result (Green + Blue)")
    o3d.visualization.draw_geometries([pcd_normals, pcd_target_viz],
                                      window_name="Normal-Aware ICP")
    
    # Show comparison
    print("4. Comparison: Basic (Red) vs Normal-aware (Green) vs Target (Blue)")
    o3d.visualization.draw_geometries([pcd_basic, pcd_normals, pcd_target_viz],
                                      window_name="ICP Methods Comparison")
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Key takeaways:")
    print("- Normal-aware correspondences typically provide better convergence")
    print("- Geometry information helps avoid incorrect matches")
    print("- Point-to-plane distances are more meaningful than Euclidean distances")
    print("- Both methods should converge to similar final transformations")
    print("="*80)

if __name__ == "__main__":
    main()
