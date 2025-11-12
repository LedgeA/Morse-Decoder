"""
Computer vision utilities for face and eye detection
"""
import math
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from ultralytics import YOLO
from config import *

class VisionProcessor:
    """`
    Handles face detection, eye tracking, and blink detection
    """
    
    def __init__(self):
        """Initialize MediaPipe face mesh and tracking variables"""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.light_model = YOLO('best.pt')

        # EAR smoothing histories
        self.left_ear_hist = deque(maxlen=EAR_SMOOTHING)
        self.right_ear_hist = deque(maxlen=EAR_SMOOTHING)
        self.avg_ear_hist = deque(maxlen=EAR_SMOOTHING)
        
        # Calibration data
        self.open_ear_values = []
        self.calibration_done = False
        self.EAR_CLOSED_THRESHOLD = None
        self.EAR_OPEN_THRESHOLD = None
        
        # Blink detection state
        self.blinking = False
        self.blink_start_time = None
        self.blink_frame_counter = 0
        self.open_frame_counter = 0
        self.blink_type = None
        
    @staticmethod
    def euclidean_distance(p1, p2) -> float:
        """
        Calculate Euclidean distance between two points
        
        Args:
            p1, p2: Points with x and y attributes
        
        Returns:
            Euclidean distance
        """
        return math.hypot(p1.x - p2.x, p1.y - p2.y)
    
    def eye_aspect_ratio(self, landmarks, eye_indices) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for given eye landmarks
        
        Args:
            landmarks: MediaPipe face landmarks
            eye_indices: Indices of eye landmarks
        
        Returns:
            EAR value (higher = more open, lower = more closed)
        """
        # Vertical distances
        A = self.euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        B = self.euclidean_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        
        # Horizontal distance
        C = self.euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
        
        if C == 0:
            return 0.0
        
        return (A + B) / (2.0 * C)
    
    def process_eye_frame(self, frame):
        """
        Process frame to detect face and calculate EAR values
        
        Args:
            frame: Input video frame
        
        Returns:
            tuple: (results, left_ear_smoothed, right_ear_smoothed, avg_ear_smoothed)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        left_ear_s, right_ear_s, avg_ear_s = 0.0, 0.0, 0.0
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate EAR for both eyes
            left_ear = self.eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = self.eye_aspect_ratio(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Apply smoothing
            self.left_ear_hist.append(left_ear)
            self.right_ear_hist.append(right_ear)
            self.avg_ear_hist.append(avg_ear)
            
            left_ear_s = sum(self.left_ear_hist) / len(self.left_ear_hist)
            right_ear_s = sum(self.right_ear_hist) / len(self.right_ear_hist)
            avg_ear_s = sum(self.avg_ear_hist) / len(self.avg_ear_hist)
            
            # Calibration phase
            if not self.calibration_done:
                self.open_ear_values.append(avg_ear_s)
        
        return results, left_ear_s, right_ear_s, avg_ear_s

    def process_light_frame(self, display_frame):
        results = self.light_model(display_frame)
        result = results[0]
        if len(result.boxes) > 0:
            box = result.boxes[0]
            # confidence = float(box.conf[0])

            return box
        return None

    def complete_calibration(self):
        """
        Complete calibration phase and set EAR thresholds
        """
        if self.open_ear_values:
            baseline_open_ear = float(np.percentile(self.open_ear_values, 80))
            self.EAR_CLOSED_THRESHOLD = baseline_open_ear * EAR_OPEN_FACTOR
            self.EAR_OPEN_THRESHOLD = self.EAR_CLOSED_THRESHOLD * EAR_HYSTERESIS_FACTOR
            self.calibration_done = True
    
    def draw_eye_landmarks(self, frame, landmarks):
        """
        Draw eye landmarks on the frame
        
        Args:
            frame: Frame to draw on
            landmarks: MediaPipe face landmarks
        """
        h, w, _ = frame.shape
        
        for eye_indices in [LEFT_EYE, RIGHT_EYE]:
            points = []
            for idx in eye_indices:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Draw eye contours
            for i in range(len(points)):
                cv2.line(frame, points[i], points[(i + 1) % len(points)], (0, 255, 0), 1)

    def detect_eye_blink(self, avg_ear_s: float, current_symbol: str, current_time: float) -> tuple:
        """
        Detect blink based on EAR values and current symbol

        Args:
            avg_ear_s: Smoothed average EAR value
            current_symbol: Current active symbol ('.', '-', or ' ')
            current_time: Current timestamp

        Returns:
            tuple: (blink_detected, new_symbol, blink_ended)
        """
        if not self.calibration_done:
            return False, None, False

        is_both_closed_raw = avg_ear_s < self.EAR_CLOSED_THRESHOLD
        is_open_raw = avg_ear_s > self.EAR_OPEN_THRESHOLD

        potential_blink_type = current_symbol if is_both_closed_raw else None

        blink_detected = False
        new_symbol = None
        blink_ended = False

        # Blink start detection
        if not self.blinking:
            if potential_blink_type == current_symbol:
                self.blink_frame_counter += 1
                self.open_frame_counter = 0
                if self.blink_frame_counter >= MIN_FRAMES_TO_BLINK:
                    self.blinking = True
                    self.blink_start_time = current_time
                    self.blink_type = potential_blink_type
                    blink_detected = True
            else:
                self.blink_frame_counter = 0

        # Blink end detection
        if self.blinking:
            if is_open_raw:
                self.open_frame_counter += 1
                self.blink_frame_counter = 0
                if self.open_frame_counter >= MIN_FRAMES_TO_OPEN:
                    blink_duration = current_time - self.blink_start_time

                    if MIN_BLINK_DURATION <= blink_duration <= MAX_BLINK_DURATION:
                        if self.blink_type in ['.', '-']:
                            new_symbol = self.blink_type
                        elif self.blink_type == ' ':
                            # SPACE DETECTED - return special flag
                            new_symbol = 'SPACE'  # Special flag for space

                    blink_ended = True
                    self._reset_blink_state()

            elif self.blink_start_time and (current_time - self.blink_start_time > MAX_BLINK_DURATION):
                # Blink too long, reset
                self._reset_blink_state()
                blink_ended = True

        return blink_detected, new_symbol, blink_ended

    def detect_light_blink(self, light_conf: float, current_symbol: str, current_time: float) -> tuple:

        is_closed_raw = light_conf < 0.5
        is_open_raw = light_conf > 0.5

        potential_blink_type = current_symbol if is_closed_raw else None

        blink_detected = False
        new_symbol = None
        blink_ended = False

        # Blink start detection
        if not self.blinking:
            if potential_blink_type == current_symbol:
                self.blink_frame_counter += 1
                self.open_frame_counter = 0
                if self.blink_frame_counter >= MIN_FRAMES_TO_BLINK:
                    self.blinking = True
                    self.blink_start_time = current_time
                    self.blink_type = potential_blink_type
                    blink_detected = True
            else:
                self.blink_frame_counter = 0

        # Blink end detection
        if self.blinking:
            if is_open_raw:
                self.open_frame_counter += 1
                self.blink_frame_counter = 0
                if self.open_frame_counter >= MIN_FRAMES_TO_OPEN:
                    blink_duration = current_time - self.blink_start_time

                    if MIN_BLINK_DURATION <= blink_duration <= MAX_BLINK_DURATION:
                        if self.blink_type in ['.', '-']:
                            new_symbol = self.blink_type
                        elif self.blink_type == ' ':
                            # SPACE DETECTED - return special flag
                            new_symbol = 'SPACE'  # Special flag for space

                    blink_ended = True
                    self._reset_blink_state()

            elif self.blink_start_time and (current_time - self.blink_start_time > MAX_BLINK_DURATION):
                # Blink too long, reset
                self._reset_blink_state()
                blink_ended = True

        return blink_detected, new_symbol, blink_ended

    def _reset_blink_state(self):
        """Reset blink detection state"""
        self.blinking = False
        self.blink_start_time = None
        self.blink_type = None
        self.blink_frame_counter = 0
        self.open_frame_counter = 0
    
    def cleanup(self):
        """Clean up vision resources"""
        if self.face_mesh:
            self.face_mesh.close()