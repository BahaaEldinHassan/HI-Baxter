# src/camera/realsense_camera.py

import pyrealsense2 as rs
import numpy as np
from typing import Optional, Tuple, Dict
import json
import logging
from pathlib import Path


class RealSenseCamera:
    """RealSense camera handler with configuration management"""

    def __init__(self,
                 config_path: Optional[str] = None,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30):
        """
        Initialize RealSense camera with optional config file

        Args:
            config_path: Path to camera configuration file
            width: Color/Depth frame width
            height: Color/Depth frame height
            fps: Camera frame rate
        """
        self.width = width
        self.height = height
        self.fps = fps

        # Initialize RealSense objects
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Load advanced settings if config file provided
        if config_path:
            self.load_config(config_path)

        # Configure streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Alignment object
        self.align = None

        # Pipeline profile
        self.profile = None

        # Camera intrinsics
        self.color_intrinsics = None
        self.depth_intrinsics = None

        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start the RealSense pipeline"""
        try:
            # Start streaming
            self.profile = self.pipeline.start(self.config)

            # Get device info
            device = self.profile.get_device()
            depth_sensor = device.first_depth_sensor()

            # Enable auto-exposure
            depth_sensor.set_option(rs.option.enable_auto_exposure, True)

            # Create alignment object
            self.align = rs.align(rs.stream.color)

            # Get stream profiles
            depth_profile = rs.video_stream_profile(
                self.profile.get_stream(rs.stream.depth))
            color_profile = rs.video_stream_profile(
                self.profile.get_stream(rs.stream.color))

            # Get intrinsics
            self.depth_intrinsics = depth_profile.get_intrinsics()
            self.color_intrinsics = color_profile.get_intrinsics()

            self.logger.info("RealSense camera started successfully")
            return True

        except RuntimeError as e:
            self.logger.error(f"Failed to start RealSense camera: {str(e)}")
            return False

    def stop(self):
        """Stop the RealSense pipeline"""
        try:
            self.pipeline.stop()
            self.logger.info("RealSense camera stopped")
        except RuntimeError as e:
            self.logger.error(f"Error stopping RealSense camera: {str(e)}")

    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get aligned color and depth frames

        Returns:
            Tuple of (color_frame, depth_frame) as numpy arrays
            or None if frames could not be captured
        """
        try:
            # Wait for a coherent pair of frames
            frames = self.pipeline.wait_for_frames()

            # Align depth to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                return None

            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return color_image, depth_image

        except RuntimeError as e:
            self.logger.error(f"Error capturing frames: {str(e)}")
            return None

    def get_depth_scale(self) -> float:
        """Get depth scale for converting depth values to meters"""
        depth_sensor = self.profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()

    def deproject_pixel_to_point(self,
                                 pixel: Tuple[int, int],
                                 depth: float) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates and depth to 3D point

        Args:
            pixel: (x, y) pixel coordinates
            depth: depth value in current depth units

        Returns:
            (x, y, z) 3D point coordinates in meters
        """
        return rs.rs2_deproject_pixel_to_point(
            self.depth_intrinsics,
            [pixel[0], pixel[1]],
            depth
        )

    def load_config(self, config_path: str):
        """Load JSON configuration file with camera settings"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Get advanced device
            device = self.profile.get_device() if self.profile else None
            if device:
                # Load JSON string as advanced settings
                json_string = str(config).replace("'", '\"')
                advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_string)

        except Exception as e:
            self.logger.error(f"Error loading config file: {str(e)}")

    def get_intrinsics(self) -> Dict:
        """Get camera intrinsics parameters"""
        return {
            'color': {
                'fx': self.color_intrinsics.fx,
                'fy': self.color_intrinsics.fy,
                'ppx': self.color_intrinsics.ppx,
                'ppy': self.color_intrinsics.ppy
            },
            'depth': {
                'fx': self.depth_intrinsics.fx,
                'fy': self.depth_intrinsics.fy,
                'ppx': self.depth_intrinsics.ppx,
                'ppy': self.depth_intrinsics.ppy
            }
        }


# Example usage
if __name__ == "__main__":
    import cv2

    # Initialize camera
    camera = RealSenseCamera()

    if not camera.start():
        print("Failed to start camera")
        exit(1)

    try:
        while True:
            # Get frames
            frames = camera.get_frames()
            if frames is not None:
                color_frame, depth_frame = frames

                # Display frames
                cv2.imshow('Color', color_frame)
                cv2.imshow('Depth', depth_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        camera.stop()
        cv2.destroyAllWindows()