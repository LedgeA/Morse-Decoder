"""
User interface utilities for displaying information
"""
import cv2
import numpy as np
import time  # Added missing import
from config import *

class UIHandler:
    """
    Handles user interface elements and display
    """
    
    def __init__(self, window_title: str = "Eye Blink Morse Code"):
        """
        Initialize UI handler
        
        Args:
            window_title: Title for the display window
        """
        self.window_title = window_title
        self.notification_message = ""
        self.notification_time = 0.0
        
        # Create window
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, 1600, 900)
    
    def show_notification(self, message: str, current_time: float, is_error: bool = False):
        """
        Show temporary notification message
        
        Args:
            message: Notification text
            current_time: Current timestamp
            is_error: Whether this is an error message
        """
        self.notification_message = message
        self.notification_time = current_time
    
    def should_show_notification(self, current_time: float) -> bool:
        """
        Check if notification should still be displayed
        
        Args:
            current_time: Current timestamp
        
        Returns:
            True if notification should be shown
        """
        return (self.notification_message and 
                (current_time - self.notification_time < NOTIFICATION_DURATION))
    
    def create_text_panel(self, height: int, morse_sequence: str, decoded_text: str, 
                         invalid_sequence: bool = False) -> np.ndarray:
        """
        Create side panel with text information
        
        Args:
            height: Panel height
            morse_sequence: Current Morse sequence
            decoded_text: Decoded text
            invalid_sequence: Whether current sequence is invalid
        
        Returns:
            Text panel image
        """
        # Create white panel
        text_panel = np.ones((height, TEXT_PANEL_WIDTH, 3), dtype=np.uint8) * 255
        
        # Header
        cv2.putText(text_panel, "MORSE INPUT (Sequential)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        cv2.line(text_panel, (10, 40), (TEXT_PANEL_WIDTH - 10, 40), (200, 200, 200), 1)
        
        # Current Morse sequence
        cv2.putText(text_panel, "Current Sequence:", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        color = (255, 0, 0) if not invalid_sequence else (0, 0, 255)
        cv2.putText(text_panel, morse_sequence, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
        
        # Decoded text section
        cv2.putText(text_panel, "DECODED TEXT", (10, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        cv2.line(text_panel, (10, 190), (TEXT_PANEL_WIDTH - 10, 190), (200, 200, 200), 1)
        
        # Format and display decoded text with word wrapping
        lines = self._wrap_text(decoded_text, max_chars=30)
        y_offset = 230
        for line in lines:
            cv2.putText(text_panel, line, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            y_offset += 35
            if y_offset > height - 100:
                break
        
        # Controls section
        self._draw_controls(text_panel, height)
        
        # Notification
        if self.should_show_notification(time.time()):  # Now time is defined
            color = (0, 0, 255) if "ERROR" in self.notification_message else (255, 0, 0)
            cv2.putText(text_panel, self.notification_message, (10, height - 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return text_panel
    
    def _wrap_text(self, text: str, max_chars: int) -> list:
        """
        Wrap text to fit within character limit
        
        Args:
            text: Text to wrap
            max_chars: Maximum characters per line
        
        Returns:
            List of wrapped lines
        """
        if not text:
            return []
            
        lines = []
        line = ""
        
        for word in text.split(" "):
            if len(line) + len(word) + 1 > max_chars:
                lines.append(line)
                line = word
            else:
                line += (" " if line else "") + word
        
        if line:
            lines.append(line)
        
        return lines
    
    def _draw_controls(self, panel: np.ndarray, height: int):
        """
        Draw control instructions on panel
        
        Args:
            panel: Panel to draw on
            height: Panel height
        """
        # Control instructions
        controls = [
            "SUBMIT [ENTER] | RESET [R] | DELETE [D]",
            "T - SPEAK (Press to speak decoded text)",
            "B - BEEP (Play Morse beeps for decoded text)",
            "QUIT [Q]"
        ]
        
        colors = [
            (0, 150, 0),    # Green for main controls
            (0, 150, 150),  # Teal for speak
            (0, 100, 200),  # Blue for beep
            (0, 0, 150)     # Dark blue for quit
        ]
        
        y_positions = [height - 120, height - 90, height - 60, height - 30]
        font_sizes = [0.6, 0.6, 0.55, 0.6]
        
        for i, (control, color, y_pos, font_size) in enumerate(
            zip(controls, colors, y_positions, font_sizes)):
            
            cv2.putText(panel, control, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)
    
    def draw_sequential_ui(self, frame, is_interval_phase: bool, current_symbol: str, 
                          time_left: float, blink_detected: bool = False):
        """
        Draw sequential input UI elements on main frame
        
        Args:
            frame: Frame to draw on
            is_interval_phase: Whether in interval phase
            current_symbol: Current active symbol
            time_left: Time left for current phase
            blink_detected: Whether blink was detected
        """
        h, w, _ = frame.shape
        
        if is_interval_phase:
            # Interval phase display
            cv2.putText(frame, "INTERVAL - REST", (w // 2 - 200, h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 3)
        else:
            # Active symbol phase display
            symbol = current_symbol
            icon_text = "DOT" if symbol == "." else "DASH" if symbol == "-" else "SPACE"
            color = (0, 255, 0) if symbol == "." else (255, 255, 0) if symbol == "-" else (255, 0, 0)

            cv2.putText(frame, f"ACTIVE: {icon_text}", (w // 2 - 150, h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(frame, f"TIME LEFT: {max(0.0, time_left):.1f}s",
                        (w // 2 - 150, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            if blink_detected:
                cv2.putText(frame, f"INPUT DETECTED: {icon_text}", (w // 2 - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    def draw_light_readings(self, frame, coords):
        if coords is not None:
            x1, y1, x2, y2 = coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    def draw_ear_readings(self, frame, left_ear: float, right_ear: float, status_text: str = ""):
        """
        Draw EAR readings and status on frame

        Args:
            frame: Frame to draw on
            left_ear: Left eye EAR value
            right_ear: Right eye EAR value
            status_text: Status text to display
        """
        cv2.putText(frame, f"L-EAR: {left_ear:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"R-EAR: {right_ear:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if status_text:
            cv2.putText(frame, status_text, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def display_combined_frame(self, camera_frame: np.ndarray, text_panel: np.ndarray):
        """
        Display combined camera feed and text panel
        
        Args:
            camera_frame: Camera feed frame
            text_panel: Text panel frame
        """
        combined = cv2.hconcat([camera_frame, text_panel])
        cv2.imshow(self.window_title, combined)
    
    def cleanup(self):
        """Clean up UI resources"""
        cv2.destroyAllWindows()