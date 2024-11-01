import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp


class PoseDepthDetector:
    def __init__(self):
        self.depth_frame = None
        self.depth_image = None
        self.color_image = None
        self.images = None
        self.depth_scale = None

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2  # Use the most accurate model
        )

        # Store landmark depths
        self.landmark_depths = {}

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            if self.depth_image is not None:
                width = self.depth_image.shape[1]
                if x < width:
                    dist = self.depth_image[y, x] * self.depth_scale
                    img_copy = self.images.copy()
                    cv2.putText(img_copy, f"Distance at cursor: {dist:.2f}m",
                                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow('RealSense Pose Detection', img_copy)


def pose_depth_measurement():
    # Initialize detector
    detector = PoseDepthDetector()

    # Configure RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    detector.depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {detector.depth_scale}")

    # Create alignment object
    align = rs.align(rs.stream.color)

    # Create window and set mouse callback
    cv2.namedWindow('RealSense Pose Detection')
    cv2.setMouseCallback('RealSense Pose Detection', detector.mouse_callback)

    try:
        while True:
            # Get and align frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            detector.depth_image = np.asanyarray(depth_frame.get_data())
            detector.color_image = np.asanyarray(color_frame.get_data())

            # Process pose
            color_image_rgb = cv2.cvtColor(detector.color_image, cv2.COLOR_BGR2RGB)
            pose_results = detector.pose.process(color_image_rgb)

            # Create depth visualization
            norm_depth = cv2.normalize(detector.depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_colormap = cv2.applyColorMap(norm_depth.astype(np.uint8), cv2.COLORMAP_RAINBOW)

            # Create visualization image
            annotated_image = detector.color_image.copy()

            if pose_results.pose_landmarks:
                # Draw pose landmarks
                detector.mp_drawing.draw_landmarks(
                    annotated_image,
                    pose_results.pose_landmarks,
                    detector.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=detector.mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Get 3D positions of key points
                height, width = detector.depth_image.shape
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    px = int(landmark.x * width)
                    py = int(landmark.y * height)

                    # Ensure coordinates are within bounds
                    if 0 <= px < width and 0 <= py < height:
                        # Get depth of landmark
                        depth = detector.depth_image[py, px] * detector.depth_scale
                        detector.landmark_depths[idx] = depth

                        # Draw depth information for key points (e.g., wrists)
                        if idx in [detector.mp_pose.PoseLandmark.LEFT_WRIST.value,
                                   detector.mp_pose.PoseLandmark.RIGHT_WRIST.value]:
                            cv2.putText(annotated_image, f"{depth:.2f}m",
                                        (px, py), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (255, 255, 255), 1)

            # Add scale bar
            height, width = depth_colormap.shape[:2]
            scale_width = 30
            scale_image = np.zeros((height, scale_width, 3), dtype=np.uint8)
            for i in range(height):
                color = cv2.applyColorMap(
                    np.array([[int(255 * (1 - i / height))]], dtype=np.uint8),
                    cv2.COLORMAP_RAINBOW)[0][0]
                scale_image[i, :] = color

            # Add distance markers to scale
            max_depth = np.max(detector.depth_image) * detector.depth_scale
            for i in range(5):
                y_pos = int(height * i / 4)
                depth_value = f"{(max_depth * (4 - i) / 4):.1f}m"
                cv2.putText(scale_image, depth_value, (2, y_pos + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Stack images
            detector.images = np.hstack((annotated_image, depth_colormap, scale_image))

            # Show images
            cv2.imshow('RealSense Pose Detection', detector.images)

            # Print depths of specific landmarks (e.g., wrists)
            if detector.landmark_depths:
                left_wrist_depth = detector.landmark_depths.get(
                    detector.mp_pose.PoseLandmark.LEFT_WRIST.value, 0)
                right_wrist_depth = detector.landmark_depths.get(
                    detector.mp_pose.PoseLandmark.RIGHT_WRIST.value, 0)
                print(f"\rLeft Wrist: {left_wrist_depth:.2f}m, Right Wrist: {right_wrist_depth:.2f}m",
                      end='')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        detector.pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pose_depth_measurement()