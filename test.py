import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def main():
    # 1. Configure the Intel RealSense D415 Stream
    pipeline = rs.pipeline()
    config = rs.config()
    
    # D415 standard resolutions for optimal depth matching
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start the camera pipeline
    profile = pipeline.start(config)
    
    # Setup RealSense Filters for clean carton surfaces (Suction Safety).
    # NOTE: decimation_filter is intentionally omitted -- it halves the depth
    # resolution (640x480 -> 320x240), which then mismatches the full-res color
    # image and the color intrinsics used below, crashing create_from_color_and_depth.
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # Depth scale (metres per depth unit) -> needed so the point cloud is metric.
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    
    # Align depth map to match color image pixel coordinates
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Get camera intrinsic matrix parameters for coordinate projection
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0,       0,        1]
    ])
    dist_coeffs = np.array(intr.coeffs)

    print("[INFO] Camera streaming started. Press 'q' in the window to exit.")

    try:
        while True:
            # 2. Wait for a coherent pair of frames (Depth + Color)
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # Apply industrial filtering stack to clean up tape/cardboard surface noise
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            # Convert images to standard numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 3. Convert RealSense Frame to an Open3D Point Cloud
            # Pass intrinsic parameters to convert 2D depth pixels into a structured 6D Cloud (X,Y,Z)
            o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
            )
            o3d_depth = o3d.geometry.Image(depth_image)
            o3d_color = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth,
                depth_scale=1.0 / depth_scale,   # depth units per metre (e.g. 1000)
                depth_trunc=2.0,                 # ignore points beyond 2 m
                convert_rgb_to_intensity=False
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_intrinsics)
            
            # Downsample the cloud to process quickly in real-time
            pcd = pcd.voxel_down_sample(voxel_size=0.01)

            # 4. Segment the Pallet Bed / Conveyor using RANSAC
            if len(pcd.points) > 100:
                # Find the dominant horizontal plane (usually the pallet or top carton surface)
                plane_model, inliers = pcd.segment_plane(
                    distance_threshold=0.02, ransac_n=3, num_iterations=100
                )
                
                # Extract the points belonging exclusively to the target carton top surface
                carton_surface = pcd.select_by_index(inliers)
                
                if len(carton_surface.points) > 50:
                    # 5. Boundary/Edge Fitting: Generate an Oriented Bounding Box (OBB)
                    obb = carton_surface.get_oriented_bounding_box()
                    
                    # Extract 3D Translation vector [X, Y, Z] relative to camera center
                    translation = obb.center  # Format: [X, Y, Z] in meters
                    
                    # Extract 3D Rotation matrix and convert to Euler angles (Roll, Pitch, Yaw)
                    rotation_matrix = obb.R
                    r = R.from_matrix(rotation_matrix)
                    euler_angles = r.as_euler('xyz', degrees=True)  # Format: [Roll, Pitch, Yaw]
                    
                    # 6. Reproject the 6D Pose Data back onto the Real-Time OpenCV Video Frame
                    # Use standard OpenCV projection to draw a physical coordinate axis on the carton
                    axis_points_3d = np.array([
                        [0, 0, 0],       # Origin (Carton Centroid)
                        [0.1, 0, 0],     # X-Axis (Red)
                        [0, 0.1, 0],     # Y-Axis (Green)
                        [0, 0, 0.1]      # Z-Axis (Blue - Pointing straight up for suction)
                    ], dtype=np.float32)

                    # Transform axis points by the box's orientation matrix and translation
                    transformed_axis = (rotation_matrix @ axis_points_3d.T).T + translation
                    
                    # Project the 3D spatial points into 2D camera pixels
                    img_points, _ = cv2.projectPoints(
                        transformed_axis, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
                    )
                    img_points = img_points.astype(int).squeeze()

                    # Draw the 3D axes on the video feed
                    cv2.line(color_image, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 3) # X
                    cv2.line(color_image, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 3) # Y
                    cv2.line(color_image, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 3) # Z

                    # Overlay the numeric 6D Pose data metrics onto the video window
                    pose_txt_pos = f"XYZ(m): [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]"
                    pose_txt_rot = f"RPY(deg): [{euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}]"
                    
                    cv2.putText(color_image, pose_txt_pos, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(color_image, pose_txt_rot, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Mark the exact center spot intended for suction engagement
                    cv2.circle(color_image, tuple(img_points[0]), 7, (0, 255, 255), -1)
                    cv2.putText(color_image, "SUCTION TARGET", (tuple(img_points[0])[0]+10, tuple(img_points[0])[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Display the real-time detection window
            cv2.imshow("RealSense D415 - Carton 6D Pose Detection", color_image)
            
            # Break loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Safely shut down streams and clean up graphic instances
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
