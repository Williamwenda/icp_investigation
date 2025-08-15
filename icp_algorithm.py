import open3d as o3d
import numpy as np
import copy
import os
from scipy.spatial.transform import Rotation as R
from icp_util import print_results

np.set_printoptions(threshold=5)

def compute_pose(roll_deg=0, pitch_deg=0, yaw_deg=0, x=0, y=0, z=0):
    """
    Compute the pose of the robot in the world frame. 
    Transformation from body frame to world frame. 
    """
    # Convert angles to radians
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    
    # Create rotation matrix (ZYX convention)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    R = R_z @ R_y @ R_x
    
    # Create transformation matrix
    T_wb = np.eye(4)
    T_wb[:3, :3] = R
    T_wb[:3, 3] = [x, y, z]

    return T_wb

def load_point_clouds():
    """
    Load the two simulated LiDAR point clouds from the current directory
    
    Returns:
        tuple: (pcd_source, pcd_target) - Source and target point clouds

    Note:
        pcd_source = lidar_scan_source.pcd (transformed scan to be aligned)
        pcd_target = lidar_scan_target.pcd (reference scan)
        The lidar scans are in lidar local frame, 
        ICP registration/alignment: transform the Source point cloud to match the Target point cloud
    """
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct full paths to the PCD files
        pcd_source_path = os.path.join(script_dir, "sim_data/lidar_scan_source.pcd")
        pcd_target_path = os.path.join(script_dir, "sim_data/lidar_scan_target.pcd")
        
        # Check if files exist
        if not os.path.exists(pcd_source_path):
            print(f"Error: {pcd_source_path} does not exist")
            return None, None
        
        if not os.path.exists(pcd_target_path):
            print(f"Error: {pcd_target_path} does not exist")
            return None, None
        
        # Load the point clouds
        pcd_source = o3d.io.read_point_cloud(pcd_source_path)
        pcd_target = o3d.io.read_point_cloud(pcd_target_path)

        return pcd_source, pcd_target
    except Exception as e:
        print(f"Error loading point clouds: {e}")
        return None, None

def preprocess_point_cloud(pcd, voxel_size=0.1):
    """
    Preprocess point cloud for ICP
    
    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for downsampling
        
    Returns:
        o3d.geometry.PointCloud: Preprocessed point cloud
    """
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals if not present
    if not pcd_down.has_normals():
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        pcd_down.normalize_normals()
    
    return pcd_down

def run_point_to_point_icp(source, target, threshold=0.2, initial_transformation=None):
    """
    Run point-to-point ICP algorithm
    
    Args:
        source: Source point cloud
        target: Target point cloud
        threshold: Distance threshold
        initial_transformation: Initial guess for transformation
        
    Returns:
        tuple: (transformation_matrix, registration_result)
    """
    if initial_transformation is None:
        initial_transformation = np.eye(4)
    
    print("Running Point-to-Point ICP...")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    print("Point-to-Point ICP transformation matrix:")
    print(reg_p2p.transformation)
    print(f"Point-to-Point ICP fitness: {reg_p2p.fitness:.6f}")
    print(f"Point-to-Point ICP inlier RMSE: {reg_p2p.inlier_rmse:.6f}")
    
    return reg_p2p.transformation, reg_p2p

def run_point_to_plane_icp(source, target, threshold=0.2, initial_transformation=None):
    """
    Run point-to-plane ICP algorithm
    
    Args:
        source: Source point cloud
        target: Target point cloud
        threshold: Distance threshold
        initial_transformation: Initial guess for transformation
        
    Returns:
        tuple: (transformation_matrix, registration_result)
    """
    if initial_transformation is None:
        initial_transformation = np.eye(4)
    
    print("Running Point-to-Plane ICP...")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    print("Point-to-Plane ICP transformation matrix:")
    print(reg_p2l.transformation)
    print(f"Point-to-Plane ICP fitness: {reg_p2l.fitness:.6f}")
    print(f"Point-to-Plane ICP inlier RMSE: {reg_p2l.inlier_rmse:.6f}")
    
    return reg_p2l.transformation, reg_p2l

def visualize_registration_result(source, target, transformation):
    """
    Visualize the registration result
    
    Args:
        source: Source point cloud (to be transformed)
        target: Target point cloud (reference)
        transformation: Transformation matrix from ICP
    """
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    
    # Color point clouds
    source_transformed.paint_uniform_color([1, 0, 0])  # Red for transformed source
    target.paint_uniform_color([0, 0, 1])  # Blue for target
    
    print("Visualizing registration result...")
    print("Red: Transformed source point cloud (lidar_scan_source.pcd aligned)")
    print("Blue: Target point cloud (lidar_scan_target.pcd reference)")
    
    o3d.visualization.draw_geometries([source_transformed, target],
                                      window_name="ICP Registration Result")

def visualize_all_icp_results(pcd_source, pcd_target, 
                              T_ground_truth, T_initial_guess, results):
    """
    Visualize all ICP results: initial guess, point-to-point ICP, and point-to-plane ICP
    
    Args:
        pcd_source: Source point cloud (to be aligned)
        pcd_target: Target point cloud (reference)
        initial_guess: Initial guess transformation matrix
        results: Results from ICP algorithms containing transformations
    """
    # Apply transformations for visualization
    source_initial = copy.deepcopy(pcd_source)
    source_initial.transform(T_initial_guess)
    
    source_gt = copy.deepcopy(pcd_source)
    source_gt.transform(T_ground_truth)

    source_p2p = copy.deepcopy(pcd_source)
    source_p2p.transform(results['point_to_point']['transformation'])
    
    source_p2l = copy.deepcopy(pcd_source)
    source_p2l.transform(results['point_to_plane']['transformation'])
    
    # Color point clouds
    source_initial.paint_uniform_color([0, 1, 0])  # Green for initial guess
    source_gt.paint_uniform_color([0.5, 0, 0.5])   # Purple for ground truth
    source_p2p.paint_uniform_color([1, 0, 0])      # Red for Point-to-Point ICP
    source_p2l.paint_uniform_color([1, 1, 0])      # Yellow for Point-to-Plane ICP
    pcd_target.paint_uniform_color([0.0, 0.0, 1.0]) # Blue for target

    print("Visualizing all ICP results...")
    print("Blue: Target point cloud (reference)")

    # o3d.visualization.draw_geometries([source_initial, source_p2p, source_p2l, pcd_target],
    #                                   window_name="ICP Results Comparison")
    
    # visualize the registration results with initial guess
    print("Visualizing initial guess alignment (Green)...")
    o3d.visualization.draw_geometries([source_initial, pcd_target],
                                      window_name="ICP Results Comparison")
    # visualize the registration results with point-to-point ICP
    print("Visualizing point-to-point ICP alignment (Red)...")
    o3d.visualization.draw_geometries([source_p2p, pcd_target],
                                      window_name="ICP Results Comparison")
    # visualize the registration results with point-to-plane ICP
    print("Visualizing point-to-plane ICP alignment (Yellow)...")
    o3d.visualization.draw_geometries([source_p2l, pcd_target],
                                      window_name="ICP Results Comparison")
    # visualize the registration results with GT transformation
    print("Visualizing GT alignment (Purple)...")
    o3d.visualization.draw_geometries([source_gt, pcd_target],
                                      window_name="ICP Results Comparison")

def compute_se3_pose_error(T_estimated, T_ground_truth):
    """
    Compute SE(3) pose error between estimated and ground truth transformations
    
    Args:
        T_estimated: 4x4 estimated transformation matrix
        T_ground_truth: 4x4 ground truth transformation matrix
        
    Returns:
        dict: Error metrics including translation and rotation errors
    """
    # Compute relative error: T_error = T_ground_truth^(-1) * T_estimated
    T_gt_inv = np.linalg.inv(T_ground_truth)
    T_error = T_gt_inv @ T_estimated
    
    # Extract translation error
    translation_error = T_error[:3, 3]
    translation_error_norm = np.linalg.norm(translation_error)
    
    # Extract rotation error using scipy
    R_error = T_error[:3, :3]
    rotation_scipy = R.from_matrix(R_error)
    
    # Get rotation error in axis-angle representation
    rotation_vector = rotation_scipy.as_rotvec()
    rotation_error_norm = np.linalg.norm(rotation_vector)
    rotation_error_deg = np.rad2deg(rotation_error_norm)
    
    # Get Euler angles for detailed analysis
    euler_error = rotation_scipy.as_euler('xyz', degrees=True)
    
    # Individual component errors (absolute differences)
    t_est = T_estimated[:3, 3]
    t_gt = T_ground_truth[:3, 3]
    R_est = R.from_matrix(T_estimated[:3, :3])
    R_gt = R.from_matrix(T_ground_truth[:3, :3])
    
    euler_est = R_est.as_euler('xyz', degrees=True)
    euler_gt = R_gt.as_euler('xyz', degrees=True)
    
    return {
        'transformation_error_matrix': T_error,
        'translation': {
            'error_vector': translation_error,
            'error_norm': translation_error_norm,
            'dx_error': abs(t_est[0] - t_gt[0]),
            'dy_error': abs(t_est[1] - t_gt[1]),
            'dz_error': abs(t_est[2] - t_gt[2])
        },
        'rotation': {
            'error_vector': rotation_vector,
            'error_norm_rad': rotation_error_norm,
            'error_norm_deg': rotation_error_deg,
            'euler_error': euler_error,
            'roll_error': abs(euler_est[0] - euler_gt[0]),
            'pitch_error': abs(euler_est[1] - euler_gt[1]),
            'yaw_error': abs(euler_est[2] - euler_gt[2])
        },
        'estimated_pose': {
            'translation': t_est,
            'euler_deg': euler_est
        },
        'ground_truth_pose': {
            'translation': t_gt,
            'euler_deg': euler_gt
        }
    }

def create_perturbation_transformation(dx, dy, dz, droll_deg, dpitch_deg, dyaw_deg):
    """
    Create a transformation matrix from perturbation values
    
    Args:
        dx, dy, dz: Translation perturbations in meters
        droll_deg, dpitch_deg, dyaw_deg: Rotation perturbations in degrees
        
    Returns:
        np.ndarray: 4x4 perturbation transformation matrix
    """
    # Convert angles to radians
    droll_rad = np.deg2rad(droll_deg)
    dpitch_rad = np.deg2rad(dpitch_deg)
    dyaw_rad = np.deg2rad(dyaw_deg)
    
    # Create rotation matrix from Euler angles (extrinsic convention in Z-Y-X order)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(droll_rad), -np.sin(droll_rad)],
                    [0, np.sin(droll_rad), np.cos(droll_rad)]])
    R_y = np.array([[np.cos(dpitch_rad), 0, np.sin(dpitch_rad)],
                    [0, 1, 0],
                    [-np.sin(dpitch_rad), 0, np.cos(dpitch_rad)]])
    R_z = np.array([[np.cos(dyaw_rad), -np.sin(dyaw_rad), 0],
                    [np.sin(dyaw_rad), np.cos(dyaw_rad), 0],
                    [0, 0, 1]])
    rotation = R_z @ R_y @ R_x

            
    # Create perturbation transformation matrix
    T_perturbation = np.eye(4)
    T_perturbation[:3, :3] = rotation
    T_perturbation[:3, 3] = [dx, dy, dz]
    
    return T_perturbation

def generate_perturbed_initial_guess_from_pose(ground_truth_matrix, 
                                               dx=0.0, dy=0.0, dz=0.0,
                                               droll_deg=0.0, dpitch_deg=0.0, dyaw_deg=0.0):
    """
    Generate an initial guess by adding specific pose perturbations to ground truth
    
    Args:
        ground_truth_matrix: 4x4 ground truth transformation matrix
        dx, dy, dz: Translation perturbations in meters
        droll_deg, dpitch_deg, dyaw_deg: Rotation perturbations in degrees
        
    Returns:
        np.ndarray: 4x4 perturbed transformation matrix as initial guess
    """
    # Create perturbation transformation matrix
    T_perturbation = create_perturbation_transformation(dx, dy, dz, droll_deg, dpitch_deg, dyaw_deg)
    
    # Apply perturbation: T_perturbed = T_perturbation * T_ground_truth
    perturbed_matrix = T_perturbation @ ground_truth_matrix
    
    return perturbed_matrix

if __name__ == "__main__":
    # set random seed for reproducible results
    np.random.seed(42)

    # lidar poses in sim
    lidar_pose1 = {
        'x': 0.0, 'y':0.0, 'z':1.5,
        'roll_deg': 0.0, 'pitch_deg': 0.0, 'yaw_deg': 0.0
    } 

    lidar_pose2 = {
        'x': 0.2, 'y': 0.5, 'z': 1.5,
        'roll_deg': 0.0, 'pitch_deg': 0.0, 'yaw_deg': 20.0
    }
    print(f"\n{'='*50}")
    print("LiDAR Pose 1, (T_w_lidar1): ")
    T_lidar_source = compute_pose(**lidar_pose1)
    print(T_lidar_source)
    T_lidar_target = compute_pose(**lidar_pose2)
    print("LiDAR Pose 2, (T_w_lidar2): ")
    print(T_lidar_target)
    print("The ground truth incremental pose change T_delta is: ")
    T_delta = T_lidar_target @ np.linalg.inv(T_lidar_source)
    print(T_delta)
    print("ICP results transform SOURCE to TARGET: \nT_lidar2_lidar1 = T_w_lidar2^(-1) @ T_w_lidar1\n")
    print("The ground truth of ICP results (transformation from lidar1 to lidar2) is: ")
    T_ground_truth = np.linalg.inv(T_lidar_target) @ T_lidar_source
    print(T_ground_truth)
    print(f"{'='*50}")

    # Option 1: Use explicit perturbation values [dx, dy, dz, droll, dpitch, dyaw]
    print(f"\n{'='*50}")
    print("GENERATING PERTURBED INITIAL GUESS (EXPLICIT VALUES)")
    print(f"{'='*50}")
    
    # Define specific perturbation values
    perturbation_values = {
        'dx': 0.5,          
        'dy': 0.0,          
        'dz': 0.0,          
        'droll_deg': 0.0,    
        'dpitch_deg': 0.0,  
        'dyaw_deg': 0.0      
    }
    T_initial_guess = generate_perturbed_initial_guess_from_pose(
        T_ground_truth, 
        dx=perturbation_values['dx'],
        dy=perturbation_values['dy'],
        dz=perturbation_values['dz'],
        droll_deg=perturbation_values['droll_deg'],
        dpitch_deg=perturbation_values['dpitch_deg'],
        dyaw_deg=perturbation_values['dyaw_deg']
    )
    print("The perturbed T_initial_guess is: ")
    print(T_initial_guess)

    # Load point clouds
    pcd_source, pcd_target = load_point_clouds()
    if pcd_source is None or pcd_target is None:
        print("Failed to load point clouds.")

    # Preprocess point clouds
    voxel_size = 0.1  # Voxel size for downsampling
    pcd_source_processed = preprocess_point_cloud(pcd_source, voxel_size)
    pcd_target_processed = preprocess_point_cloud(pcd_target, voxel_size)


    # ICP algorithm param
    threshold = 0.2
    # Point-to-Point ICP (source -> target)
    print(f"\nRunning ICP algorithms with threshold={threshold}m...")
    T_p2p, reg_p2p = run_point_to_point_icp(pcd_source_processed, pcd_target_processed, threshold, T_initial_guess)
    errors_p2p = compute_se3_pose_error(T_p2p, T_ground_truth)
    
    # Point-to-Plane ICP (source -> target)
    T_p2l, reg_p2l = run_point_to_plane_icp(pcd_source_processed, pcd_target_processed, threshold, T_initial_guess)
    errors_p2l = compute_se3_pose_error(T_p2l, T_ground_truth)

    # Package results
    results = {
        'point_to_point': {
            'transformation': T_p2p,
            'registration_result': reg_p2p,
            'errors': errors_p2p
        },
        'point_to_plane': {
            'transformation': T_p2l,
            'registration_result': reg_p2l,
            'errors': errors_p2l
        },
        'T_ground_truth': T_ground_truth,
        'T_initial_guess': T_initial_guess,
        'parameters': {
            'threshold': threshold,
            'voxel_size': voxel_size
        }
    }

    print_results(results)

    # Visualize all ICP results
    visualize_all_icp_results(pcd_source_processed, pcd_target_processed,
                              T_ground_truth, T_initial_guess, results)