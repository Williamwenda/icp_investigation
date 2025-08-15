#!/usr/bin/env python3
"""
ICP Implementation from Scratch

This module implements a point-to-plane ICP algorithm from scratch using only 
NumPy and SciPy for core computations. 
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import svd
import time


class ICPResult:
    """Container for ICP results, similar to Open3D's RegistrationResult."""
    
    def __init__(self, transformation=None, fitness=0.0, inlier_rmse=0.0, 
                 correspondence_set=None, num_iterations=0):
        self.transformation = transformation if transformation is not None else np.eye(4)
        self.fitness = fitness
        self.inlier_rmse = inlier_rmse
        self.correspondence_set = correspondence_set if correspondence_set is not None else []
        self.num_iterations = num_iterations


def estimate_normals_simple(points, k=20):
    """
    Estimate normals for point cloud using PCA on local neighborhoods.
    
    Args:
        points (np.ndarray): Point cloud of shape (N, 3)
        k (int): Number of neighbors for normal estimation
        
    Returns:
        np.ndarray: Normals of shape (N, 3)
    """
    if len(points) < k:
        k = len(points) - 1
    
    # Build KD-tree for efficient neighbor search
    tree = KDTree(points)
    normals = np.zeros_like(points)
    
    for i, point in enumerate(points):
        # Find k nearest neighbors
        distances, indices = tree.query(point, k=k+1)  # +1 to include the point itself
        neighbors = points[indices[1:]]  # Exclude the point itself
        
        if len(neighbors) < 3:
            # Not enough neighbors for reliable normal estimation
            normals[i] = np.array([0, 0, 1])  # Default upward normal
            continue
        
        # Center the neighbors
        centroid = np.mean(neighbors, axis=0)
        centered = neighbors - centroid
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered.T)
        
        # Find normal as eigenvector with smallest eigenvalue
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        
        # Ensure consistent orientation (point normals away from centroid)
        if np.dot(normal, point - centroid) < 0:
            normal = -normal
            
        normals[i] = normal / np.linalg.norm(normal)
    
    return normals


def find_correspondences(source_points, target_points, max_distance=0.5):
    """
    Find point correspondences between source and target point clouds.
    
    Args:
        source_points (np.ndarray): Source points of shape (N, 3)
        target_points (np.ndarray): Target points of shape (M, 3)
        max_distance (float): Maximum distance for valid correspondences
        
    Returns:
        tuple: (source_indices, target_indices, distances)
    """
    # Build KD-tree for target points
    tree = KDTree(target_points)
    
    # Find nearest neighbors for each source point
    distances, target_indices = tree.query(source_points)
    
    # Filter correspondences by distance threshold
    valid_mask = distances < max_distance
    source_indices = np.where(valid_mask)[0]
    target_indices = target_indices[valid_mask]
    distances = distances[valid_mask]
    
    return source_indices, target_indices, distances


def find_correspondences_with_normals(source_points, target_points, 
                                     source_normals, target_normals,
                                     max_distance=0.5, max_normal_angle=25.0):
    """
    Find correspondences using both spatial and normal information.
    Enhanced version that considers normal compatibility for better matching.
    
    Args:
        source_points (np.ndarray): Source points of shape (N, 3)
        target_points (np.ndarray): Target points of shape (M, 3)
        source_normals (np.ndarray): Source normals of shape (N, 3)
        target_normals (np.ndarray): Target normals of shape (M, 3)
        max_distance (float): Maximum distance for valid correspondences
        max_normal_angle (float): Maximum angle between normals in degrees
        
    Returns:
        tuple: (source_indices, target_indices, distances) - same format as find_correspondences
    """
    tree = KDTree(target_points)
    
    source_indices = []
    target_indices = []
    distances = []
    
    for i, (src_pt, src_normal) in enumerate(zip(source_points, source_normals)):
        # Get K nearest candidates for normal-aware selection
        k_candidates = min(5, len(target_points))
        dists, indices = tree.query(src_pt, k=k_candidates)
        
        # Handle single result case
        if np.isscalar(dists):
            dists = [dists]
            indices = [indices]
        
        best_match = None
        best_score = float('inf')
        
        for dist, idx in zip(dists, indices):
            if dist > max_distance:
                continue
                
            tgt_normal = target_normals[idx]
            
            # Calculate angle between normals (use absolute value for unsigned angle)
            dot_product = np.clip(np.dot(src_normal, tgt_normal), -1.0, 1.0)
            normal_angle = np.degrees(np.arccos(np.abs(dot_product)))
            
            # Reject if normals are too different
            if normal_angle > max_normal_angle:
                continue
            
            # Calculate point-to-plane distance for more accurate scoring
            point_diff = src_pt - target_points[idx]
            point_to_plane_dist = abs(np.dot(point_diff, tgt_normal))
            
            # Combined score: emphasize point-to-plane distance
            normalized_dist = dist / max_distance
            normalized_angle = normal_angle / max_normal_angle
            score = point_to_plane_dist + 0.1 * normalized_dist + 0.01 * normalized_angle
            
            if score < best_score:
                best_score = score
                best_match = (i, idx, dist)
        
        if best_match:
            source_indices.append(best_match[0])
            target_indices.append(best_match[1])
            distances.append(best_match[2])
    
    return np.array(source_indices), np.array(target_indices), np.array(distances)


def estimate_transformation_point_to_plane(source_points, target_points, target_normals, 
                                         source_indices, target_indices):
    """
    Estimate rigid transformation using point-to-plane error minimization.
    
    This implements the linearized point-to-plane ICP algorithm that solves
    for the transformation parameters directly using least squares.
    
    Args:
        source_points (np.ndarray): Source points of shape (N, 3)
        target_points (np.ndarray): Target points of shape (M, 3)
        target_normals (np.ndarray): Target normals of shape (M, 3)
        source_indices (np.ndarray): Valid source point indices
        target_indices (np.ndarray): Corresponding target point indices
        
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    if len(source_indices) < 6:
        # Not enough correspondences for 6-DOF estimation
        return np.eye(4)
    
    # Get corresponding points and normals
    p_s = source_points[source_indices]  # Source points
    p_t = target_points[target_indices]  # Target points
    n_t = target_normals[target_indices]  # Target normals
    
    # Build the linear system for point-to-plane ICP
    # We want to solve: A * x = b
    # where x = [alpha, beta, gamma, tx, ty, tz]^T represents rotation angles and translation
    
    n_corr = len(p_s)
    A = np.zeros((n_corr, 6))
    b = np.zeros(n_corr)
    
    for i in range(n_corr):
        ps = p_s[i]
        pt = p_t[i]
        nt = n_t[i]
        
        # Cross product terms for rotation (linearized rotation matrix)
        cross_x = np.array([0, -ps[2], ps[1]])
        cross_y = np.array([ps[2], 0, -ps[0]])
        cross_z = np.array([-ps[1], ps[0], 0])
        
        # Fill the A matrix
        A[i, 0] = np.dot(cross_x, nt)  # Rotation around x-axis
        A[i, 1] = np.dot(cross_y, nt)  # Rotation around y-axis
        A[i, 2] = np.dot(cross_z, nt)  # Rotation around z-axis
        A[i, 3] = nt[0]                # Translation x
        A[i, 4] = nt[1]                # Translation y
        A[i, 5] = nt[2]                # Translation z
        
        # Fill the b vector (residual)
        b[i] = np.dot(nt, pt - ps)
    
    # Solve the linear system using least squares
    try:
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.eye(4)
    
    # Extract rotation angles and translation
    alpha, beta, gamma = x[:3]  # Rotation angles (small angles approximation)
    translation = x[3:6]
    
    # Build rotation matrix from small angles (first-order approximation)
    R = np.array([[1, -gamma, beta],
                  [gamma, 1, -alpha],
                  [-beta, alpha, 1]])
    
    # Ensure rotation matrix is orthogonal using SVD
    U, _, Vt = svd(R)
    R = U @ Vt
    
    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    
    return T


def compute_fitness_and_rmse(source_points, target_points, target_normals,
                           source_indices, target_indices, transformation, 
                           max_distance=0.5):
    """
    Compute fitness and inlier RMSE for ICP evaluation.
    
    Args:
        source_points (np.ndarray): Source points
        target_points (np.ndarray): Target points  
        target_normals (np.ndarray): Target normals
        source_indices (np.ndarray): Valid source indices
        target_indices (np.ndarray): Corresponding target indices
        transformation (np.ndarray): 4x4 transformation matrix
        max_distance (float): Distance threshold for inliers
        
    Returns:
        tuple: (fitness, inlier_rmse)
    """
    if len(source_indices) == 0:
        return 0.0, 0.0
    
    # Transform source points
    source_transformed = transform_points(source_points, transformation)
    
    # Get corresponding points
    p_s_transformed = source_transformed[source_indices]
    p_t = target_points[target_indices]
    n_t = target_normals[target_indices]
    
    # Compute point-to-plane distances
    point_to_plane_distances = np.abs(np.sum((p_s_transformed - p_t) * n_t, axis=1))
    
    # Count inliers
    inliers = point_to_plane_distances < max_distance
    num_inliers = np.sum(inliers)
    
    # Compute fitness (fraction of inliers)
    fitness = num_inliers / len(source_points)
    
    # Compute RMSE for inliers only
    if num_inliers > 0:
        inlier_rmse = np.sqrt(np.mean(point_to_plane_distances[inliers] ** 2))
    else:
        inlier_rmse = float('inf')
    
    return fitness, inlier_rmse


def transform_points(points, transformation):
    """
    Transform points using a 4x4 transformation matrix.
    
    Args:
        points (np.ndarray): Points of shape (N, 3)
        transformation (np.ndarray): 4x4 transformation matrix
        
    Returns:
        np.ndarray: Transformed points of shape (N, 3)
    """
    # Convert to homogeneous coordinates
    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
    
    # Apply transformation
    transformed_homogeneous = (transformation @ points_homogeneous.T).T
    
    # Convert back to 3D coordinates
    return transformed_homogeneous[:, :3]


def icp_point_to_plane(source_points, target_points, source_normals=None, target_normals=None,
                       initial_guess=None, max_iterations=100, tolerance=1e-6, 
                       max_correspondence_distance=0.5, verbose=False):
    """
    Point-to-Plane ICP implementation from scratch.
    
    This function implements the iterative closest point algorithm using point-to-plane
    error metric, similar to Open3D's registration_icp with TransformationEstimationPointToPlane.
    
    Args:
        source_points (np.ndarray): Source point cloud of shape (N, 3)
        target_points (np.ndarray): Target point cloud of shape (M, 3)
        source_normals (np.ndarray, optional): Source normals of shape (N, 3)
        target_normals (np.ndarray, optional): Target normals of shape (M, 3)
        initial_guess (np.ndarray, optional): Initial 4x4 transformation matrix
        max_iterations (int): Maximum number of ICP iterations
        tolerance (float): Convergence tolerance for transformation change
        max_correspondence_distance (float): Maximum distance for valid correspondences
        verbose (bool): Print iteration details
        
    Returns:
        ICPResult: Object containing transformation, fitness, RMSE, and other metrics
    """
    if verbose:
        print(f"Starting Point-to-Plane ICP with {len(source_points)} source and {len(target_points)} target points")
    
    # Initialize transformation
    if initial_guess is None:
        current_transformation = np.eye(4)
    else:
        current_transformation = initial_guess.copy()
    
    # Estimate normals if not provided
    if target_normals is None:
        if verbose:
            print("Estimating target normals...")
        target_normals = estimate_normals_simple(target_points)
    
    if source_normals is None:
        if verbose:
            print("Estimating source normals...")
        source_normals = estimate_normals_simple(source_points)

    
    # Initialize variables for iteration
    prev_error = float('inf')
    best_transformation = current_transformation.copy()
    best_fitness = 0.0
    best_rmse = float('inf')
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Transform source points with current estimate
        source_transformed = transform_points(source_points, current_transformation)
        
        # Find correspondences
        source_indices, target_indices, distances = find_correspondences(
            source_transformed, target_points, max_correspondence_distance)
        


        if len(source_indices) < 6:
            if verbose:
                print(f"Iteration {iteration}: Insufficient correspondences ({len(source_indices)})")
            break
        
        # Estimate transformation increment
        delta_transformation = estimate_transformation_point_to_plane(
            source_transformed, target_points, target_normals, 
            source_indices, target_indices)
        
        # Update total transformation
        current_transformation = delta_transformation @ current_transformation
        
        # Compute fitness and RMSE
        fitness, rmse = compute_fitness_and_rmse(
            source_points, target_points, target_normals,
            source_indices, target_indices, current_transformation,
            max_correspondence_distance)
        
        # Check for improvement
        if fitness > best_fitness or (fitness == best_fitness and rmse < best_rmse):
            best_transformation = current_transformation.copy()
            best_fitness = fitness
            best_rmse = rmse
        
        # Check convergence
        transformation_change = np.linalg.norm(delta_transformation - np.eye(4))
        
        if verbose:
            print(f"Iteration {iteration:2d}: correspondences={len(source_indices):4d}, "
                  f"fitness={fitness:.4f}, RMSE={rmse:.6f}, change={transformation_change:.2e}")
        
        if transformation_change < tolerance:
            if verbose:
                print(f"Converged after {iteration + 1} iterations")
            break
        
        prev_error = rmse
    
    else:
        if verbose:
            print(f"Reached maximum iterations ({max_iterations})")
    
    # Final correspondence set for the best transformation
    source_final = transform_points(source_points, best_transformation)
    source_indices, target_indices, distances = find_correspondences(
        source_final, target_points, max_correspondence_distance)
    
    correspondence_set = list(zip(source_indices, target_indices))
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"ICP completed in {elapsed_time:.3f} seconds")
        print(f"Final: fitness={best_fitness:.6f}, inlier_rmse={best_rmse:.6f}")
    
    return ICPResult(
        transformation=best_transformation,
        fitness=best_fitness,
        inlier_rmse=best_rmse,
        correspondence_set=correspondence_set,
        num_iterations=iteration + 1
    )


def compare_correspondence_methods(source_points, target_points, source_normals, target_normals, 
                                  max_distance=0.5, max_normal_angle=25.0, verbose=True):
    """
    Compare the two correspondence methods side by side.
    
    Args:
        source_points: Source point coordinates
        target_points: Target point coordinates  
        source_normals: Source point normals
        target_normals: Target point normals
        max_distance: Distance threshold
        max_normal_angle: Normal angle threshold in degrees
        verbose: Print detailed comparison
        
    Returns:
        dict: Comparison results
    """
    if verbose:
        print("="*60)
        print("CORRESPONDENCE METHODS COMPARISON")
        print("="*60)
    
    # Method 1: Basic Euclidean correspondence
    start_time = time.time()
    src_idx_basic, tgt_idx_basic, dist_basic = find_correspondences(
        source_points, target_points, max_distance)
    time_basic = time.time() - start_time
    
    # Method 2: Normal-aware correspondence
    start_time = time.time()
    src_idx_normals, tgt_idx_normals, dist_normals = find_correspondences_with_normals(
        source_points, target_points, source_normals, target_normals, 
        max_distance, max_normal_angle)
    time_normals = time.time() - start_time
    
    # Calculate statistics
    results = {
        'basic': {
            'num_correspondences': len(src_idx_basic),
            'mean_distance': np.mean(dist_basic) if len(dist_basic) > 0 else 0,
            'std_distance': np.std(dist_basic) if len(dist_basic) > 0 else 0,
            'time': time_basic,
            'source_indices': src_idx_basic,
            'target_indices': tgt_idx_basic,
            'distances': dist_basic
        },
        'normals': {
            'num_correspondences': len(src_idx_normals),
            'mean_distance': np.mean(dist_normals) if len(dist_normals) > 0 else 0,
            'std_distance': np.std(dist_normals) if len(dist_normals) > 0 else 0,
            'time': time_normals,
            'source_indices': src_idx_normals,
            'target_indices': tgt_idx_normals,
            'distances': dist_normals
        }
    }
    
    if verbose:
        print(f"{'Method':<20} {'Correspondences':<15} {'Mean Dist':<12} {'Std Dist':<12} {'Time (ms)':<10}")
        print("-" * 75)
        print(f"{'Basic Euclidean':<20} {results['basic']['num_correspondences']:<15} "
              f"{results['basic']['mean_distance']:<12.4f} {results['basic']['std_distance']:<12.4f} "
              f"{results['basic']['time']*1000:<10.2f}")
        print(f"{'Normal-Aware':<20} {results['normals']['num_correspondences']:<15} "
              f"{results['normals']['mean_distance']:<12.4f} {results['normals']['std_distance']:<12.4f} "
              f"{results['normals']['time']*1000:<10.2f}")
        
        # Quality analysis
        print(f"\nQuality Analysis:")
        if results['normals']['num_correspondences'] > 0 and results['basic']['num_correspondences'] > 0:
            ratio = results['normals']['num_correspondences'] / results['basic']['num_correspondences']
            print(f"- Correspondence retention: {ratio:.2%}")
            
            distance_improvement = (results['basic']['mean_distance'] - results['normals']['mean_distance']) / results['basic']['mean_distance'] * 100
            print(f"- Distance improvement: {distance_improvement:.1f}%")
        
        print("="*60)
    
    return results


def icp_point_to_plane_with_method_choice(source_points, target_points, source_normals=None, target_normals=None,
                                        initial_guess=None, max_iterations=100, tolerance=1e-6, 
                                        max_correspondence_distance=0.5, use_normals=True, verbose=False):
    """
    Point-to-Plane ICP implementation that allows choosing correspondence method.
    
    Args:
        source_points (np.ndarray): Source point cloud of shape (N, 3)
        target_points (np.ndarray): Target point cloud of shape (M, 3)
        source_normals (np.ndarray, optional): Source normals of shape (N, 3)
        target_normals (np.ndarray, optional): Target normals of shape (M, 3)
        initial_guess (np.ndarray, optional): Initial 4x4 transformation matrix
        max_iterations (int): Maximum number of ICP iterations
        tolerance (float): Convergence tolerance for transformation change
        max_correspondence_distance (float): Maximum distance for valid correspondences
        use_normals (bool): Whether to use normal-aware correspondence (if normals available)
        verbose (bool): Print iteration details
        
    Returns:
        ICPResult: Object containing transformation, fitness, RMSE, and other metrics
    """
    if verbose:
        method_name = "Normal-Aware" if (use_normals and source_normals is not None and target_normals is not None) else "Basic Euclidean"
        print(f"Starting Point-to-Plane ICP with {method_name} correspondences...")
        print(f"Source points: {len(source_points)}, Target points: {len(target_points)}")
    
    # Initialize transformation
    if initial_guess is None:
        current_transformation = np.eye(4)
    else:
        current_transformation = initial_guess.copy()
    
    # Check if we can and should use normals
    can_use_normals = (source_normals is not None and target_normals is not None)
    will_use_normals = use_normals and can_use_normals
    
    if verbose and use_normals and not can_use_normals:
        print("Warning: Normal-aware correspondence requested but normals not available")
    
    # Initialize variables for iteration
    prev_error = float('inf')
    best_transformation = current_transformation.copy()
    best_fitness = 0.0
    best_rmse = float('inf')
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Transform source points with current estimate
        source_transformed = transform_points(source_points, current_transformation)
        
        # Transform source normals if available and using normals
        source_normals_transformed = None
        if will_use_normals:
            R = current_transformation[:3, :3]
            source_normals_transformed = (R @ source_normals.T).T
            # Normalize to ensure unit vectors
            norms = np.linalg.norm(source_normals_transformed, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            source_normals_transformed = source_normals_transformed / norms
        
        # Find correspondences using chosen method
        if will_use_normals:
            source_indices, target_indices, distances = find_correspondences_with_normals(
                source_transformed, target_points, source_normals_transformed, target_normals,
                max_correspondence_distance, max_normal_angle=60.0)
        else:
            source_indices, target_indices, distances = find_correspondences(
                source_transformed, target_points, max_correspondence_distance)
        
        if len(source_indices) < 6:
            if verbose:
                print(f"Iteration {iteration}: Insufficient correspondences ({len(source_indices)})")
            break
        
        # Estimate transformation increment
        delta_transformation = estimate_transformation_point_to_plane(
            source_transformed, target_points, target_normals, 
            source_indices, target_indices)
        
        # Update total transformation
        current_transformation = delta_transformation @ current_transformation
        
        # Compute fitness and RMSE
        fitness, rmse = compute_fitness_and_rmse(
            source_points, target_points, target_normals,
            source_indices, target_indices, current_transformation,
            max_correspondence_distance)
        
        # Check for improvement
        if fitness > best_fitness or (fitness == best_fitness and rmse < best_rmse):
            best_transformation = current_transformation.copy()
            best_fitness = fitness
            best_rmse = rmse
        
        # Check convergence
        transformation_change = np.linalg.norm(delta_transformation - np.eye(4))
        
        if verbose:
            method_str = "N" if will_use_normals else "E"
            print(f"Iteration {iteration:2d} [{method_str}]: correspondences={len(source_indices):4d}, "
                  f"fitness={fitness:.4f}, RMSE={rmse:.6f}, change={transformation_change:.2e}")
        
        if transformation_change < tolerance:
            if verbose:
                print(f"Converged after {iteration + 1} iterations")
            break
        
        prev_error = rmse
    
    else:
        if verbose:
            print(f"Reached maximum iterations ({max_iterations})")
    
    # Final correspondence set for the best transformation
    source_final = transform_points(source_points, best_transformation)
    if will_use_normals:
        R = best_transformation[:3, :3]
        source_normals_final = (R @ source_normals.T).T
        norms = np.linalg.norm(source_normals_final, axis=1, keepdims=True)
        source_normals_final = source_normals_final / norms
        source_indices, target_indices, distances = find_correspondences_with_normals(
            source_final, target_points, source_normals_final, target_normals,
            max_correspondence_distance, max_normal_angle=60.0)
    else:
        source_indices, target_indices, distances = find_correspondences(
            source_final, target_points, max_correspondence_distance)
    
    correspondence_set = list(zip(source_indices, target_indices))
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"ICP completed in {elapsed_time:.3f} seconds")
        print(f"Final: fitness={best_fitness:.6f}, inlier_rmse={best_rmse:.6f}")
        print(f"Method used: {'Normal-Aware' if will_use_normals else 'Basic Euclidean'}")
    
    return ICPResult(
        transformation=best_transformation,
        fitness=best_fitness,
        inlier_rmse=best_rmse,
        correspondence_set=correspondence_set,
        num_iterations=iteration + 1
    )


if __name__ == "__main__":
    # Simple test demonstrating both correspondence methods
    print("ICP from scratch implementation with dual correspondence methods!")
    print("="*70)
    print("Available functions:")
    print("1. find_correspondences() - Basic Euclidean nearest neighbor")
    print("2. find_correspondences_with_normals() - Normal-aware correspondence")
    print("3. compare_correspondence_methods() - Compare both methods")
    print("4. icp_point_to_plane() - Original ICP implementation")
    print("5. icp_point_to_plane_with_method_choice() - ICP with method selection")
    print("="*70)
    print("Import this module and use these functions in your pointcloud.py script.")
    print("Example usage:")
    print("  from icp_scratch import compare_correspondence_methods")
    print("  results = compare_correspondence_methods(src_pts, tgt_pts, src_norms, tgt_norms)")
    print("="*70)


def preprocess_point_cloud_simple(points, voxel_size=0.1):
    """
    Simple point cloud preprocessing: voxel downsampling.
    
    Args:
        points (np.ndarray): Input points of shape (N, 3)
        voxel_size (float): Voxel size for downsampling
        
    Returns:
        np.ndarray: Downsampled points
    """
    if voxel_size <= 0:
        return points
    
    # Simple voxel grid downsampling
    min_coords = np.min(points, axis=0)
    voxel_indices = np.floor((points - min_coords) / voxel_size).astype(int)
    
    # Get unique voxels
    unique_voxels, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    
    return points[unique_indices]



