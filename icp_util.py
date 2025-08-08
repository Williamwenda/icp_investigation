import open3d as o3d
import numpy as np
import copy
import os
from scipy.spatial.transform import Rotation as R

def print_detailed_results(algorithm_name, registration_result, errors):
    """
    Print detailed ICP results with ground truth comparison
    
    Args:
        algorithm_name: Name of the ICP algorithm
        registration_result: Open3D registration result
        errors: Error metrics dictionary
    """
    print(f"\n{'='*60}")
    print(f"{algorithm_name.upper()} RESULTS")
    print(f"{'='*60}")
    
    # Registration quality metrics
    print(f"Registration Quality:")
    print(f"  Fitness: {registration_result.fitness:.4f}")
    print(f"  RMSE: {registration_result.inlier_rmse:.4f} meters")
    
    # Estimated pose
    est_pose = errors['estimated_pose']
    print(f"\nEstimated Pose:")
    print(f"  Translation: dx={est_pose['translation'][0]:.3f}m, "
          f"dy={est_pose['translation'][1]:.3f}m, dz={est_pose['translation'][2]:.3f}m")
    print(f"  Rotation: roll={est_pose['euler_deg'][0]:.1f}°, "
          f"pitch={est_pose['euler_deg'][1]:.1f}°, yaw={est_pose['euler_deg'][2]:.1f}°")
    
    # Ground truth
    gt_pose = errors['ground_truth_pose']
    print(f"\nGround Truth:")
    print(f"  Translation: dx={gt_pose['translation'][0]:.3f}m, "
          f"dy={gt_pose['translation'][1]:.3f}m, dz={gt_pose['translation'][2]:.3f}m")
    print(f"  Rotation: roll={gt_pose['euler_deg'][0]:.1f}°, "
          f"pitch={gt_pose['euler_deg'][1]:.1f}°, yaw={gt_pose['euler_deg'][2]:.1f}°")
    
    # Errors
    print(f"\nErrors:")
    print(f"  Translation errors: dx={errors['translation']['dx_error']:.3f}m, "
          f"dy={errors['translation']['dy_error']:.3f}m, dz={errors['translation']['dz_error']:.3f}m")
    print(f"  Total translation error: {errors['translation']['error_norm']:.3f}m")
    print(f"  Rotation errors: roll={errors['rotation']['roll_error']:.1f}°, "
          f"pitch={errors['rotation']['pitch_error']:.1f}°, yaw={errors['rotation']['yaw_error']:.1f}°")
    print(f"  Total rotation error: {errors['rotation']['error_norm_deg']:.1f}°")

def print_results(results):
    """
    Print comprehensive results from ICP algorithms with error analysis
    
    Args:
        results: Dictionary containing results from both ICP algorithms
    """
    print("\n" + "="*80)
    print("ICP ALGORITHM COMPARISON RESULTS")
    print("="*80)
    
    # Print detailed results for each algorithm
    print_detailed_results("Point-to-Point ICP", 
                          results['point_to_point']['registration_result'],
                          results['point_to_point']['errors'])
    
    print_detailed_results("Point-to-Plane ICP", 
                          results['point_to_plane']['registration_result'],
                          results['point_to_plane']['errors'])
    
    # Summary comparison table
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Algorithm':<20} {'Fitness':<10} {'RMSE (m)':<12} {'Trans Err (m)':<15} {'Rot Err (°)':<12}")
    print(f"{'-'*80}")
    
    p2p_errors = results['point_to_point']['errors']
    p2p_reg = results['point_to_point']['registration_result']
    p2l_errors = results['point_to_plane']['errors']
    p2l_reg = results['point_to_plane']['registration_result']
    
    print(f"{'Point-to-Point':<20} {p2p_reg.fitness:<10.4f} {p2p_reg.inlier_rmse:<12.4f} "
          f"{p2p_errors['translation']['error_norm']:<15.3f} {p2p_errors['rotation']['error_norm_deg']:<12.1f}")
    print(f"{'Point-to-Plane':<20} {p2l_reg.fitness:<10.4f} {p2l_reg.inlier_rmse:<12.4f} "
          f"{p2l_errors['translation']['error_norm']:<15.3f} {p2l_errors['rotation']['error_norm_deg']:<12.1f}")
    
    # Determine best algorithm
    p2p_total_error = p2p_errors['translation']['error_norm'] + p2p_errors['rotation']['error_norm_deg']/10
    p2l_total_error = p2l_errors['translation']['error_norm'] + p2l_errors['rotation']['error_norm_deg']/10
    
    best_algorithm = "Point-to-Point" if p2p_total_error < p2l_total_error else "Point-to-Plane"
    print(f"\nBest performing algorithm: {best_algorithm}")
    print(f"(Based on combined translation + rotation/10 error metric)")
    
    # Extrinsic calibration info
    print(f"\n{'='*80}")
    print("COORDINATE FRAME INFORMATION")
    print(f"{'='*80}")
    print("• Point clouds transformed from LiDAR frame to robot base frame")
    print("• LiDAR mounted 1.5m above robot base (Z-axis offset)")
    print("• Ground truth and estimates are in robot coordinate frame")
