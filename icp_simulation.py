'''
    3D LiDAR point cloud simulation in a mobile robot scenario.
'''
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

class LidarSimulator:
    def __init__(self, h_fov_degrees = 120, v_fov_degrees = 30,
                 max_range = 20.0, num_points = 10000):
        """
        Initialize 3D LiDAR simulator
        
        Args:
            h_fov_degrees: Horizontal field of view in degrees (default 120°)
            v_fov_degrees: Vertical field of view in degrees (default 30°)
            max_range: Maximum range in meters (default 20m)
            num_points: Target number of points to generate
        """
        self.h_fov_degrees = h_fov_degrees
        self.v_fov_degrees = v_fov_degrees
        self.max_range = max_range
        self.min_range = 2.0
        self.num_points = num_points

        # calculate horizontal and vertical resolution
        h_rays = int(np.sqrt(num_points * (h_fov_degrees / v_fov_degrees)))
        v_rays = int(num_points / h_rays)

        self.h_angular_resolution = h_fov_degrees / h_rays
        self.v_angular_resolution = v_fov_degrees / v_rays

        # generate scan angles
        self.h_angles = np.arange(-h_fov_degrees/2, h_fov_degrees/2 + self.h_angular_resolution, self.h_angular_resolution)
        self.v_angles = np.arange(-v_fov_degrees/2, v_fov_degrees/2 + self.v_angular_resolution, self.v_angular_resolution)
        
        self.h_angles_rad = np.deg2rad(self.h_angles)
        self.v_angles_rad = np.deg2rad(self.v_angles)

        # default LiDAR pose (can be changed for each scan)
        self.lidar_pose = {
            'x': 0.0, 'y': 0.0, 'z': 1.5,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
    
    def set_lidar_pose(self, x, y, z, roll_deg, pitch_deg, yaw_deg):
        """
        Set the LiDAR pose for the next scan
        
        Args:
            x, y, z: LiDAR position in meters (z defaults to lidar_height)
            roll_deg, pitch_deg, yaw_deg: LiDAR orientation in degrees
        """
        self.lidar_pose = {
            'x': x, 'y': y, 'z': z,
            'roll': np.deg2rad(roll_deg), 
            'pitch': np.deg2rad(pitch_deg), 
            'yaw': np.deg2rad(yaw_deg)
        }
        print(f"LiDAR pose set to: pos=({x:.2f}, {y:.2f}, {z:.2f}), orient=({roll_deg:.1f}°, {pitch_deg:.1f}°, {yaw_deg:.1f}°)")
    
    def scan_environment(self, cube_positions= None):
        """
        Generate a 3D lidar scan of an environment with cubes and ground plane
        The LiDAR scans from its current pose (set by set_lidar_pose)
        
        Args:
            cube_positions: List of cube positions [(x, y, z_center, size), ...]
            
        Returns:
            o3d.geometry.PointCloud: Generated point cloud in the LiDAR's local frame
        """
        # default obstacles if none provided
        if cube_positions is None:
            cube_positions = [
                (8.0, 2.0, 0.5, 1.0),   # Cube 1: x, y, z_center, size
                (12.0, -3.0, 1.0, 2.0), # Cube 2: x, y, z_center, size
            ]
        # Create LiDAR transformation matrix from current pose
        lidar_x, lidar_y, lidar_z = self.lidar_pose['x'], self.lidar_pose['y'], self.lidar_pose['z']
        roll, pitch, yaw = self.lidar_pose['roll'], self.lidar_pose['pitch'], self.lidar_pose['yaw']
        
        # Create rotation matrices
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        # Combined rotation matrix (ZYX convention)
        R_lidar = R_z @ R_y @ R_x

        points = []

        # Generate 3D rays in LiDAR's local coordinate frame
        for h_angle in self.h_angles_rad:
            for v_angle in self.v_angles_rad:
                # Ray direction in LiDAR's local frame
                ray_dir_local = np.array([
                    np.cos(v_angle) * np.cos(h_angle),  # x
                    np.cos(v_angle) * np.sin(h_angle),  # y
                    np.sin(v_angle)                     # z
                ])

                # Transform ray direction to world frame
                ray_dir_world = R_lidar @ ray_dir_local
                
                # Ray origin in world frame (LiDAR position)
                ray_origin_world = np.array([lidar_x, lidar_y, lidar_z])

                # Find intersection with environment (in world coordinates)
                hit_point_world, hit_distance = self._ray_environment_intersection(ray_origin_world, ray_dir_world, cube_positions)
                if hit_point_world is not None and self.min_range <= hit_distance <= self.max_range:
                    # Transform hit point from world coordinates to LiDAR local coordinates
                    hit_point_local = R_lidar.T @ (hit_point_world - ray_origin_world)
                    # Added noise
                    noise = np.random.normal(0, 0.01, 3)
                    points.append(hit_point_local + noise)

        # Convert points to Open3D PointCloud
        if points:
            points_array = np.array(points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_array)

            # Estimate normals
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
            pcd.normalize_normals()

            return pcd
        else:
            # return empty point cloud if no points were generated
            print("No points generated in the scan.")
            return o3d.geometry.PointCloud()
        
    def _ray_environment_intersection(self, ray_origin, ray_dir, cube_positions):
        """
        Find intersection of 3D ray with environment (ground plane and cubes)
        
        Args:
            ray_origin: Ray origin point [x, y, z]
            ray_dir: Ray direction vector [x, y, z]
            cube_positions: List of cube parameters [(x, y, z_center, size), ...]
            
        Returns:
            tuple: (hit_point, distance) or (None, None)
        """
        closest_distance = self.max_range
        closest_point = None

        # check intersection with cubes (obstacles)
        for cube in cube_positions:
            c_x, c_y, c_z, c_size = cube
            
            # Cube bounds
            half_size = c_size / 2.0
            min_x, max_x = c_x - half_size, c_x + half_size
            min_y, max_y = c_y - half_size, c_y + half_size
            min_z, max_z = c_z - half_size, c_z + half_size

            # Ray-box intersection using slab method
            hit_point, distance = self._ray_box_intersection_3d(ray_origin, ray_dir,
                                                                [min_x, min_y, min_z],
                                                                [max_x, max_y, max_z])

            if hit_point is not None and distance < closest_distance:
                closest_distance = distance
                closest_point = hit_point

        # check the ground plane if no cube was hit
        if closest_point is None:
            # check intersection with ground plane (z = 0)
            if ray_dir[2] != 0:       # ray is not parallel to ground
                t_ground = -ray_origin[2] / ray_dir[2]
                if t_ground >= 0:     # ray goes forward
                    ground_point = ray_origin + t_ground * ray_dir
                    distance = np.linalg.norm(ground_point - ray_origin)

                    # make sure ground point is not inside any cube
                    point_in_cube = False
                    for cube in cube_positions:
                        c_x, c_y, c_z, c_size = cube
                        half_size = c_size / 2.0

                        if (c_x - half_size <= ground_point[0] <= c_x + half_size and
                            c_y - half_size <= ground_point[1] <= c_y + half_size and
                            c_z - half_size <= ground_point[2] <= c_z + half_size):
                            point_in_cube = True
                            break

                    if not point_in_cube and distance < closest_distance:
                        closest_distance = distance
                        closest_point = ground_point
        
        return closest_point, closest_distance
             
    def _ray_box_intersection_3d(self, ray_origin, ray_dir, box_min, box_max):
        """
        Ray-box intersection using the slab method with surface detection
        
        Args:
            ray_origin: Ray origin [x, y, z]
            ray_dir: Ray direction [x, y, z]
            box_min: Box minimum corner [x, y, z]
            box_max: Box maximum corner [x, y, z]
            
        Returns:
            tuple: (hit_point, distance) or (None, None)
        """
        # Avoid division by zero
        ray_dir = np.where(np.abs(ray_dir) < 1e-8, 1e-8, ray_dir)
        
        # Calculate t values for each slab
        t_min = (box_min - ray_origin) / ray_dir
        t_max = (box_max - ray_origin) / ray_dir
        
        # Swap if needed to ensure t_min <= t_max
        t_enter = np.minimum(t_min, t_max)
        t_exit = np.maximum(t_min, t_max)
        
        # Find the intersection
        t_near = np.max(t_enter)
        t_far = np.min(t_exit)
        
        # Check if intersection exists and is in front of the ray
        if t_near <= t_far and t_near > 1e-6:  # Small epsilon to avoid self-intersection
            hit_point = ray_origin + t_near * ray_dir
            distance = np.linalg.norm(hit_point - ray_origin)
            
            # Determine which face was hit and filter out unwanted surfaces
            epsilon = 1e-6
            
            # Check which face was hit
            if abs(hit_point[2] - box_max[2]) < epsilon:  # Top face
                # Only allow top face hits if ray is coming from above and at a reasonable angle
                if ray_dir[2] < -0.1 and ray_origin[2] > box_max[2] + 0.1:  # Ray going down from above
                    return hit_point, distance
                else:
                    return None, None  # Filter out grazing hits on top surface
            elif abs(hit_point[2] - box_min[2]) < epsilon:  # Bottom face
                # Bottom face hits are usually ground reflections, filter them
                return None, None
            else:
                # Side faces are good
                return hit_point, distance
        
        return None, None

# visualization functions
def visualize_point_clouds_separately(pcd1_local, pcd2_local, pose_params, title1="Source Point Cloud", title2="Target Point Cloud"):
    """
    Visualize two point clouds separately with their coordinate frames
    Point clouds are in local LiDAR frames and will be transformed to world frame for visualization
    """
    import copy
    
    # Make copies to avoid modifying originals
    pcd1_vis = copy.deepcopy(pcd1_local)
    pcd2_vis = copy.deepcopy(pcd2_local)
    
    # Transform point clouds from local frames to world frame for visualization
    # pcd1 was taken from pose 1 (origin), pcd2 was taken from transformed pose
    
    # Create transformation matrix for pose 1 (reference pose - typically origin)
    # For visualization, we assume pose 1 is at origin with no rotation
    T_pose1 = np.eye(4)
    T_pose1[:3, 3] = [0, 0, 1.5]  # LiDAR height above ground
    
    # Create transformation matrix for pose 2 (transformed pose)
    roll = np.deg2rad(pose_params['roll_deg'])
    pitch = np.deg2rad(pose_params['pitch_deg']) 
    yaw = np.deg2rad(pose_params['yaw_deg'])
    
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
    
    # Create transformation matrix for pose 2
    T_pose2 = np.eye(4)
    T_pose2[:3, :3] = R
    T_pose2[:3, 3] = [pose_params['x'], pose_params['y'], pose_params['z']]
    
    # Transform point clouds to world frame
    pcd1_vis.transform(T_pose1)  # Transform from local frame to world frame
    pcd2_vis.transform(T_pose2)  # Transform from local frame to world frame
    
    # Color the point clouds differently
    pcd1_vis.paint_uniform_color([1, 0, 0])  # Red
    pcd2_vis.paint_uniform_color([0, 0, 1])  # Blue
    
    # Create coordinate frame for first pose (reference)
    coord1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    coord1.transform(T_pose1)
    
    # Create coordinate frame for second pose (transformed)
    coord2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    coord2.transform(T_pose2)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION - POINT CLOUDS IN WORLD FRAME")
    print(f"{'='*60}")
    print(f"• {title1} (Red): Scan from pose 1 transformed to world frame")
    print(f"  - {len(pcd1_local.points)} points")
    print(f"  - LiDAR pose 1: [0.0, 0.0, 1.5] m, [0°, 0°, 0°] (RPY)")
    print(f"\n• {title2} (Blue): Scan from pose 2 transformed to world frame")
    print(f"  - {len(pcd2_local.points)} points") 
    print(f"  - LiDAR pose 2: [{pose_params['x']:.3f}, {pose_params['y']:.3f}, {pose_params['z']:.3f}] m")
    print(f"  - Rotation: [{pose_params['roll_deg']:.1f}°, {pose_params['pitch_deg']:.1f}°, {pose_params['yaw_deg']:.1f}°] (RPY)")
    print(f"\nCoordinate frames show LiDAR poses:")
    print(f"• Red frame: LiDAR position during scan 1")
    print(f"• Blue frame: LiDAR position during scan 2")
    print(f"• Both scans show the same static environment from different viewpoints")
    print(f"• In local frames, these scans would appear misaligned (for ICP processing)")
    
    # Visualize both point clouds and coordinate frames together
    o3d.visualization.draw_geometries([pcd1_vis, pcd2_vis, coord1, coord2], 
                                      window_name="LiDAR Point Clouds - World Frame View",
                                      point_show_normal=False)
    
def visualize_point_clouds_individual(pcd1_local, pcd2_local, pose1, pose2, title1="Source Point Cloud", title2="Target Point Cloud"):
    """
    Visualize each point cloud individually in separate windows
    Both point clouds are transformed to world frame for proper visualization
    """
    import copy
    
    # Transform local frame point clouds to world frame
    pcd1_world = copy.deepcopy(pcd1_local)
    pcd2_world = copy.deepcopy(pcd2_local)
    
    # Create transformation matrix for pose 1
    T_pose1 = np.eye(4)
    roll1 = np.deg2rad(pose1['roll_deg'])
    pitch1 = np.deg2rad(pose1['pitch_deg'])
    yaw1 = np.deg2rad(pose1['yaw_deg'])
    
    R_x1 = np.array([[1, 0, 0],
                     [0, np.cos(roll1), -np.sin(roll1)],
                     [0, np.sin(roll1), np.cos(roll1)]])
    R_y1 = np.array([[np.cos(pitch1), 0, np.sin(pitch1)],
                     [0, 1, 0],
                     [-np.sin(pitch1), 0, np.cos(pitch1)]])
    R_z1 = np.array([[np.cos(yaw1), -np.sin(yaw1), 0],
                     [np.sin(yaw1), np.cos(yaw1), 0],
                     [0, 0, 1]])
    R1 = R_z1 @ R_y1 @ R_x1
    T_pose1[:3, :3] = R1
    T_pose1[:3, 3] = [pose1['x'], pose1['y'], pose1['z']]
    
    # Create transformation matrix for pose 2
    T_pose2 = np.eye(4)
    roll2 = np.deg2rad(pose2['roll_deg'])
    pitch2 = np.deg2rad(pose2['pitch_deg'])
    yaw2 = np.deg2rad(pose2['yaw_deg'])
    
    R_x2 = np.array([[1, 0, 0],
                     [0, np.cos(roll2), -np.sin(roll2)],
                     [0, np.sin(roll2), np.cos(roll2)]])
    R_y2 = np.array([[np.cos(pitch2), 0, np.sin(pitch2)],
                     [0, 1, 0],
                     [-np.sin(pitch2), 0, np.cos(pitch2)]])
    R_z2 = np.array([[np.cos(yaw2), -np.sin(yaw2), 0],
                     [np.sin(yaw2), np.cos(yaw2), 0],
                     [0, 0, 1]])
    R2 = R_z2 @ R_y2 @ R_x2
    T_pose2[:3, :3] = R2
    T_pose2[:3, 3] = [pose2['x'], pose2['y'], pose2['z']]
    
    # Transform to world frame
    pcd1_world.transform(T_pose1)
    pcd2_world.transform(T_pose2)
    
    # Visualize first point cloud in world frame
    pcd1_vis = copy.deepcopy(pcd1_world)
    pcd1_vis.paint_uniform_color([1, 0, 0])  # Red
    coord1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    
    # Create coordinate frame at pose 1 position
    coord_pose1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    coord_pose1.transform(T_pose1)
    
    print(f"\n{'='*40}")
    print(f"{title1.upper()} - WORLD FRAME VIEW")
    print(f"{'='*40}")
    print(f"• Points: {len(pcd1_local.points)}")
    print(f"• LiDAR pose: pos=({pose1['x']:.2f}, {pose1['y']:.2f}, {pose1['z']:.2f}), orient=({pose1['yaw_deg']:.1f}°)")
    print(f"• Color: Red")
    print(f"• Small coordinate frame shows LiDAR position")
    print("• Close this window to see the next point cloud")
    
    o3d.visualization.draw_geometries([pcd1_vis, coord1, coord_pose1], 
                                      window_name=f"{title1} - World Frame",
                                      point_show_normal=False)
    
    # Visualize second point cloud in world frame
    pcd2_vis = copy.deepcopy(pcd2_world)
    pcd2_vis.paint_uniform_color([0, 0, 1])  # Blue
    
    # Create coordinate frame at pose 2 position
    coord_pose2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    coord_pose2.transform(T_pose2)
    
    print(f"\n{'='*40}")
    print(f"{title2.upper()} - WORLD FRAME VIEW")
    print(f"{'='*40}")
    print(f"• Points: {len(pcd2_local.points)}")
    print(f"• LiDAR pose: pos=({pose2['x']:.2f}, {pose2['y']:.2f}, {pose2['z']:.2f}), orient=({pose2['yaw_deg']:.1f}°)")
    print(f"• Color: Blue")
    print(f"• Small coordinate frame shows LiDAR position")
    print("• Both scans show the same static environment from different LiDAR positions")
    
    o3d.visualization.draw_geometries([pcd2_vis, coord1, coord_pose2], 
                                      window_name=f"{title2} - World Frame",
                                      point_show_normal=False)

def visualize_point_clouds(pcd1, pcd2, title1="Source Point Cloud", title2="Target Point Cloud"):
    """
    Visualize two point clouds side by side (legacy function for compatibility)
    """
    # Color the point clouds differently
    pcd1.paint_uniform_color([1, 0, 0])  # Red
    pcd2.paint_uniform_color([0, 0, 1])  # Blue
    
    # Create coordinate frames
    coord1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    
    # Visualize
    print(f"Visualizing {title1} (Red) and {title2} (Blue)")
    print(f"{title1}: {len(pcd1.points)} points")
    print(f"{title2}: {len(pcd2.points)} points")
    
    o3d.visualization.draw_geometries([pcd1, pcd2, coord1], 
                                      window_name="LiDAR Point Clouds Comparison",
                                      point_show_normal=False)

def visualize_point_clouds_split_view(pcd1_local, pcd2_local, pose1, pose2, 
                                      title1="Source Point Cloud", title2="Target Point Cloud", 
                                      show_coordinate_frames=True, coordinate_frame_size=1.0):
    """
    Visualize two point clouds in a single window with vertical separation (one above, one below)
    Point clouds are transformed from local frame to world frame for proper visualization
    
    Args:
        pcd1_local, pcd2_local: Point clouds in local LiDAR frames
        pose1, pose2: Absolute poses for transforming to world frame
        title1, title2: Titles for the point clouds
        show_coordinate_frames: Whether to show coordinate frames (small axes)
        coordinate_frame_size: Size of the coordinate frames in meters
    """
    import copy
    
    # Transform point clouds to world frame first
    pcd1_world = copy.deepcopy(pcd1_local)
    pcd2_world = copy.deepcopy(pcd2_local)
    
    # Create transformation matrices for both poses
    def create_transform_matrix(pose):
        T = np.eye(4)
        roll = np.deg2rad(pose['roll_deg'])
        pitch = np.deg2rad(pose['pitch_deg'])
        yaw = np.deg2rad(pose['yaw_deg'])
        
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
        T[:3, :3] = R
        T[:3, 3] = [pose['x'], pose['y'], pose['z']]
        return T
    
    # Transform to world frame
    T_pose1 = create_transform_matrix(pose1)
    T_pose2 = create_transform_matrix(pose2)
    pcd1_world.transform(T_pose1)
    pcd2_world.transform(T_pose2)
    
    # Make copies for visualization
    pcd1_vis = copy.deepcopy(pcd1_world)
    pcd2_vis = copy.deepcopy(pcd2_world)
    
    # Color the point clouds differently
    pcd1_vis.paint_uniform_color([1, 0, 0])  # Red
    pcd2_vis.paint_uniform_color([0, 0, 1])  # Blue
    
    # Calculate separation distance for split view
    separation_distance = 5
    
    # Move the second point cloud upward to create vertical separation
    translation_offset = np.array([0, 0, separation_distance])
    pcd2_vis.translate(translation_offset)
    
    # List to hold all geometries for visualization
    geometries = [pcd1_vis, pcd2_vis]
    
    # Optionally create coordinate frames (the small axes you see)
    if show_coordinate_frames:
        # These are the small colored axes that show the orientation of each scan
        # Red = X-axis, Green = Y-axis, Blue = Z-axis
        coord1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_frame_size, origin=[0, 0, 0])
        coord2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_frame_size, origin=[0, 0, 0])
        
        # Transform coordinate frames to show LiDAR poses
        coord1.transform(T_pose1)
        coord2_displaced = copy.deepcopy(coord2)
        coord2_displaced.transform(T_pose2)
        coord2_displaced.translate(translation_offset)  # Also move the coordinate frame up
        
        # Add coordinate frames to visualization
        geometries.extend([coord1, coord2_displaced])
    
    print(f"\n{'='*60}")
    print("VISUALIZATION - SPLIT VIEW (UPPER/LOWER) - WORLD FRAME")
    print(f"{'='*60}")
    print(f"• LOWER VIEW - {title1} (Red): Scan from pose 1")
    print(f"  - {len(pcd1_local.points)} points")
    print(f"  - LiDAR pose 1: pos=({pose1['x']:.2f}, {pose1['y']:.2f}, {pose1['z']:.2f}), orient=({pose1['yaw_deg']:.1f}°)")
    print(f"  - Position: Original Z-level")
    
    print(f"\n• UPPER VIEW - {title2} (Blue): Scan from pose 2")
    print(f"  - {len(pcd2_local.points)} points") 
    print(f"  - LiDAR pose 2: pos=({pose2['x']:.2f}, {pose2['y']:.2f}, {pose2['z']:.2f}), orient=({pose2['yaw_deg']:.1f}°)")
    print(f"  - Position: Elevated by {separation_distance:.1f}m for visualization")
    
    if show_coordinate_frames:
        print(f"\n• COORDINATE FRAMES (Small Axes):")
        print(f"  - Red axis = X-direction, Green axis = Y-direction, Blue axis = Z-direction")
        print(f"  - Lower frame: Shows reference orientation")
        print(f"  - Upper frame: Shows transformed orientation (rotated by RPY angles)")
        print(f"  - Frame size: {coordinate_frame_size:.1f}m")
        print(f"  - Useful for understanding the relative rotation between poses")
    else:
        print(f"\n• Coordinate frames disabled for cleaner view")
    
    print(f"\nVisualization Notes:")
    print(f"• Both point clouds shown in same window, vertically separated by {separation_distance:.1f}m")
    print(f"• Use mouse to rotate and examine both poses")
    print(f"• This helps visualize the relative transformation for ICP algorithm testing")
    
    # Combine all geometries for visualization
    o3d.visualization.draw_geometries(geometries, 
                                      window_name=f"LiDAR Point Clouds - Split View ({title1} vs {title2})",
                                      point_show_normal=False)

def analyze_dominant_ratio(pcd, name="Point Cloud"):
    """
    Analyze the dominant ratio of the point cloud
    """
    if len(pcd.normals) == 0:
        print(f"{name}: No normals available")
        return None
    
    # Get normal vectors
    normals = np.asarray(pcd.normals)
    
    # Compute scatter matrix
    scatter = np.zeros((3, 3))
    for n in normals:
        n = n.reshape(3, 1)
        scatter += n @ n.T
    
    # Eigenvalue decomposition
    eigenvalues, _ = np.linalg.eigh(scatter)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    
    # Calculate dominant ratio
    total_variance = np.sum(eigenvalues)
    dominant_ratio = eigenvalues[0] / total_variance
    
    print(f"{name} Analysis:")
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Dominant ratio: {dominant_ratio:.3f}")
    print(f"  Surface type: {'Flat-like' if dominant_ratio > 0.8 else '3D geometry'}")
    
    return dominant_ratio

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

if __name__ == "__main__":

    print("3D LiDAR Point Cloud Generation with Robot Navigation")
    print("=" * 60)

    # Define static environment (cubes remain in fixed world positions)
    cube_positions = [
        (14.0, 5.0, 1.0, 2.0),   # Cube 1
        (17.0, -6.0, 1.0, 2.0),  # Cube 2 
        (10.0, 10.0, 1.5, 3.0),  # Cube 3
    ]
     
    # define lidar poses
    lidar_pose1 = {
        'x': 0.0, 'y':0.0, 'z':1.5,
        'roll_deg': 0.0, 'pitch_deg': 0.0, 'yaw_deg': 0.0
    } 

    lidar_pose2 = {
        'x': 0.2, 'y': 0.5, 'z': 1.5,
        'roll_deg': 0.0, 'pitch_deg': 0.0, 'yaw_deg': 20.0
    }

    print("LiDAR Pose 1: ")
    T_lidar1 = compute_pose(**lidar_pose1)
    print(T_lidar1)
    T_lidar2 = compute_pose(**lidar_pose2)
    print("LiDAR Pose 2: ")
    print(T_lidar2)
    print("The ground truth incremental pose change is: ")
    delta_pose = T_lidar2 @ np.linalg.inv(T_lidar1)
    print(delta_pose)


    # Create 3D LiDAR simulator
    lidar = LidarSimulator(h_fov_degrees=120, v_fov_degrees=30, max_range=20.0, num_points=50000)
    print("\nSimulating realistic robot navigation scenario:")

    # Generate first point cloud: LiDAR at pose 1
    print("Scan 1: LiDAR scanning from lidar pose 1...")
    lidar.set_lidar_pose(**lidar_pose1)
    pcd_source = lidar.scan_environment(cube_positions=cube_positions)
    
    # Generate second point cloud: LiDAR at pose 2
    print("Scan 2: LiDAR scanning from lidar pose 2...")
    lidar.set_lidar_pose(**lidar_pose2)
    pcd_target = lidar.scan_environment(cube_positions=cube_positions)

    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pcd_source_path = os.path.join(current_dir, "lidar_scan_source.pcd")  # Source scan
    pcd_target_path = os.path.join(current_dir, "lidar_scan_target.pcd")  # Target scan

    o3d.io.write_point_cloud(pcd_source_path, pcd_source)
    o3d.io.write_point_cloud(pcd_target_path, pcd_target)
    print(f"\nPoint clouds saved to:")
    print(f"  {pcd_source_path} (source scan from lidar pose 1)")
    print(f"  {pcd_target_path} (target scan from lidar pose 2)")


    # Analyze dominant ratios
    print("\nDominant Ratio Analysis:")
    print("-" * 30)
    ratio1 = analyze_dominant_ratio(pcd_source, "Scan 1 (from lidar pose 1)")
    ratio2 = analyze_dominant_ratio(pcd_target, "Scan 2 (from lidar pose 2)")

    # Ask user for visualization preference
    print(f"\n{'='*60}")    
    print("VISUALIZATION OPTIONS")
    print(f"{'='*60}")
    print("Choose visualization mode:")
    print("1. Both poses together (default)")
    print("2. Individual poses (separate windows)")
    print("3. Split view (upper/lower in one window)")
    print("4. All options")
    
    try:
        choice = input("\nEnter choice (1/2/3/4) [default=4]: ").strip()
        if choice == '':
            choice = '3'
    except:
        choice = '3'
    
    if choice in ['1', '4']:
        # Visualize both poses together with separate coordinate frames
        # Pass pose2 parameters (pose1 is assumed to be origin)
        visualize_point_clouds_separately(pcd_source, pcd_target, lidar_pose2, 
                                        "LiDAR Scan 1 (Pose 1)", "LiDAR Scan 2 (Pose 2)")
    
    if choice in ['2', '4']:
        # Visualize individual poses
        visualize_point_clouds_individual(pcd_source, pcd_target, lidar_pose1, lidar_pose2,
                                        "LiDAR Scan 1 (Pose 1)", "LiDAR Scan 2 (Pose 2)")
    
    if choice in ['3', '4']:
        # Visualize split view (upper/lower) with configurable coordinate frames
        visualize_point_clouds_split_view(pcd_source, pcd_target, lidar_pose1, lidar_pose2,
                                        "LiDAR Scan 1 (Pose 1)", "LiDAR Scan 2 (Pose 2)",
                                        show_coordinate_frames=True,  # Set to False to hide small axes
                                        coordinate_frame_size=1.0)    # Smaller coordinate frames
