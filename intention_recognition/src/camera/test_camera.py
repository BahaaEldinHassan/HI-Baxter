import pyrealsense2 as rs
import numpy as np
import cv2


class DistanceMeasurement:
    def __init__(self):
        self.depth_frame = None
        self.depth_image = None
        self.color_image = None
        self.images = None
        self.depth_scale = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # Check if we have valid depth data
            if self.depth_image is not None:
                # Check if mouse is in the color image or depth image half
                width = self.depth_image.shape[1]
                if x < width:  # If in color image
                    dist = self.depth_image[y, x] * self.depth_scale
                    # Create a copy of the image to avoid modifying the original
                    img_copy = self.images.copy()
                    # Add distance text for mouse position
                    cv2.putText(img_copy, f"Distance at cursor: {dist:.2f}m",
                                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow('RealSense', img_copy)


def enhanced_distance_measurement():
    # Initialize the measurement class
    dist_measure = DistanceMeasurement()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    dist_measure.depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {dist_measure.depth_scale}")

    # Create alignment object
    align = rs.align(rs.stream.color)

    # Create window and set mouse callback
    cv2.namedWindow('RealSense')
    cv2.setMouseCallback('RealSense', dist_measure.mouse_callback)

    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()

            # Align frames
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            dist_measure.depth_image = np.asanyarray(depth_frame.get_data())
            dist_measure.color_image = np.asanyarray(color_frame.get_data())

            # Create a normalized depth image for better visualization
            norm_depth = cv2.normalize(dist_measure.depth_image, None, 0, 255, cv2.NORM_MINMAX)

            # Create a custom colormap for better depth visualization
            depth_colormap = cv2.applyColorMap(norm_depth.astype(np.uint8), cv2.COLORMAP_RAINBOW)

            # Add a color scale bar
            height, width = depth_colormap.shape[:2]
            scale_width = 30
            scale_image = np.zeros((height, scale_width, 3), dtype=np.uint8)
            for i in range(height):
                color = \
                cv2.applyColorMap(np.array([[int(255 * (1 - i / height))]], dtype=np.uint8), cv2.COLORMAP_RAINBOW)[0][0]
                scale_image[i, :] = color

            # Add distance markers to the scale
            max_depth = np.max(dist_measure.depth_image) * dist_measure.depth_scale
            for i in range(5):
                y_pos = int(height * i / 4)
                depth_value = f"{(max_depth * (4 - i) / 4):.1f}m"
                cv2.putText(scale_image, depth_value, (2, y_pos + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Stack images horizontally with scale bar
            dist_measure.images = np.hstack((dist_measure.color_image, depth_colormap, scale_image))

            # Get distance at center of image
            height, width = dist_measure.depth_image.shape
            center_x, center_y = width // 2, height // 2
            center_dist = dist_measure.depth_image[center_y, center_x] * dist_measure.depth_scale

            # Draw crosshair at center
            cv2.line(dist_measure.images, (center_x - 10, center_y),
                     (center_x + 10, center_y), (0, 255, 0), 2)
            cv2.line(dist_measure.images, (center_x, center_y - 10),
                     (center_x, center_y + 10), (0, 255, 0), 2)

            # Add text for center distance
            cv2.putText(dist_measure.images,
                        f"Center Distance: {center_dist:.2f}m",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2)

            # Show images
            cv2.imshow('RealSense', dist_measure.images)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    enhanced_distance_measurement()