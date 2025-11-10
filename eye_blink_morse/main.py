"""
Main application file for Eye Blink Morse Code system
This is the central file that coordinates all the other modules
"""
import time
import cv2
from config import *
from morse_utils import *
from audio_utils import AudioManager
from vision_utils import VisionProcessor
from ui_utils import UIHandler

class EyeBlinkMorseApp:
    """
    Main application class for blink-based Morse code input
    This class acts as the brain that connects camera, vision, audio, and UI
    """
    
    def __init__(self):
        """Initialize all the different parts of our application"""
        # Create managers for different functions
        self.audio_manager = AudioManager()        # Handles sounds and speech
        self.vision_processor = VisionProcessor()  # Handles camera and eye detection
        self.ui_handler = UIHandler("Eye Blink Morse Code (Sequential Input - Both Eyes Only)")  # Handles display
        
        # Track what Morse code the user has entered so far
        self.morse_sequence = ""      # Current sequence of dots and dashes (like ".-.")
        self.decoded_text = ""        # Translated text (like "A")
        self.invalid_sequence_detected = False  # Flag for invalid Morse codes
        
        # Control the sequential input system (DOT -> DASH -> SPACE cycling)
        self.sequence_index = 0       # Which symbol we're currently on
        self.sequence_start_time = 0.0  # When the current symbol started
        self.current_symbol_data = INPUT_SEQUENCE[0]  # Current symbol (. or - or space)
        self.is_interval_phase = False  # Are we in rest period between symbols?
        
        # Track calibration progress
        self.frame_count = 0          # How many frames we've processed
        self.calibration_done = False  # Has calibration finished?
        
        # Prevent too many rapid key presses
        self.last_key_time = 0
        
        # Camera object (will be initialized later)
        self.cap = None
    
    def initialize_camera(self):
        """Set up the camera with the right settings"""
        self.cap = cv2.VideoCapture(camera_index)
        
        # Configure camera resolution and speed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        # Make sure camera opened successfully
        if not self.cap.isOpened():
            raise Exception("Error: Could not open video stream.")
        
        # Print actual camera speed for debugging
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Attempted to set FPS to {TARGET_FPS}. Actual FPS: {actual_fps}")
    
    def handle_key_input(self, key: int, current_time: float):
        """
        Process keyboard commands from user
        Returns True to keep running, False to quit
        """
        # Ignore keys if pressed too quickly (prevent double-presses)
        if current_time - self.last_key_time <= KEY_COOLDOWN:
            return True
        
        # Check which key was pressed and handle it
        if key == ord('q') or key == ord('Q'):
            return False  # Quit the application
        
        elif key in [ord('r'), ord('R')]:
            self._handle_reset(current_time)  # Clear current Morse sequence
        
        elif key == 13:  # Enter key
            self._handle_submit(current_time)  # Convert Morse to letter
        
        elif key in [ord('d'), ord('D')]:
            self._handle_delete(current_time)  # Delete last character
        
        elif key in [ord('t'), ord('T')]:
            self._handle_speak(current_time)   # Speak the decoded text
        
        elif key in [ord('b'), ord('B')]:
            self._handle_beep(current_time)    # Play Morse code sounds
        
        # Keep running unless 'Q' was pressed
        return True
    
    def _handle_reset(self, current_time: float):
        """Clear the current Morse sequence and reset state"""
        self.morse_sequence = ""
        self.invalid_sequence_detected = False
        self.ui_handler.show_notification("RESET: Current Morse sequence cleared.", current_time)
        print("RESET: Current Morse sequence cleared.")
        self.last_key_time = current_time
    
    def _handle_submit(self, current_time: float):
        """Convert current Morse sequence to a letter and add to decoded text"""
        if self.morse_sequence:
            # Try to translate Morse code to a letter/number
            letter = morse_to_letter(self.morse_sequence)
            if letter:
                # Success! Add the letter to our decoded text
                self.decoded_text += letter
                message = f"SUBMITTED: '{self.morse_sequence}' decoded to '{letter}'"
                self.ui_handler.show_notification(message, current_time)
                print(message)
            else:
                # Morse sequence doesn't match any known code
                message = f"ERROR: '{self.morse_sequence}' is not valid."
                self.ui_handler.show_notification(message, current_time, is_error=True)
                print(message)
            
            # Reset for next input
            self.morse_sequence = ""
            self.invalid_sequence_detected = False
        else:
            # User pressed Enter with no Morse sequence
            message = "ERROR: No Morse sequence to submit."
            self.ui_handler.show_notification(message, current_time, is_error=True)
        
        self.last_key_time = current_time
    
    def _handle_delete(self, current_time: float):
        """Remove the last character from the decoded text"""
        if self.decoded_text:
            deleted_char = self.decoded_text[-1]
            self.decoded_text = self.decoded_text[:-1]  # Remove last character
            message = f"DELETED: Removed '{deleted_char}'."
            self.ui_handler.show_notification(message, current_time)
            print(message)
        else:
            message = "ERROR: Nothing to delete."
            self.ui_handler.show_notification(message, current_time, is_error=True)
        
        self.last_key_time = current_time
    
    def _handle_speak(self, current_time: float):
        """Use text-to-speech to read the decoded text aloud"""
        if self.decoded_text:
            self.ui_handler.show_notification("Speaking decoded text...", current_time)
            self.audio_manager.speak_text_non_blocking(self.decoded_text)
        else:
            message = "ERROR: No text to speak."
            self.ui_handler.show_notification(message, current_time, is_error=True)
        
        self.last_key_time = current_time
    
    def _handle_beep(self, current_time: float):
        """Play the decoded text as Morse code beeps"""
        if self.decoded_text:
            message = f"Playing Morse beeps for: '{self.decoded_text}'"
            self.ui_handler.show_notification(message, current_time)
            self.audio_manager.play_text_as_morse_non_blocking(self.decoded_text)
        else:
            message = "ERROR: No decoded text to play."
            self.ui_handler.show_notification(message, current_time, is_error=True)
        
        self.last_key_time = current_time
    
    def update_sequential_input(self, current_time: float):
        """
        Cycle through the input sequence: DOT -> rest -> DASH -> rest -> SPACE -> rest
        This creates the timed interface for blink input
        """
        if not self.calibration_done:
            return  # Don't start until calibration is complete
        
        # Calculate how long current phase has been active
        elapsed_time = current_time - self.sequence_start_time
        
        # Determine how long this phase should last
        duration = INTERVAL_DURATION if self.is_interval_phase else self.current_symbol_data["duration"]
        
        # Check if it's time to switch to next phase
        if elapsed_time > duration:
            if self.is_interval_phase:
                # Move to next symbol in sequence
                self.sequence_index = (self.sequence_index + 1) % len(INPUT_SEQUENCE)
                self.current_symbol_data = INPUT_SEQUENCE[self.sequence_index]
                self.is_interval_phase = False  # Switch to symbol phase
            else:
                # Switch to rest period between symbols
                self.is_interval_phase = True
            
            # Reset timer for new phase
            self.sequence_start_time = current_time
    
    def process_blink_input(self, avg_ear_s: float, current_time: float):
        """
        Detect when user blinks and add the corresponding Morse symbol
        avg_ear_s is the Eye Aspect Ratio - lower means eyes are more closed
        """
        # Only process blinks during symbol phases (not during rest periods)
        if not self.is_interval_phase and self.calibration_done:
            symbol = self.current_symbol_data["symbol"]  # Current active symbol
            
            # Ask vision processor to detect if this is a valid blink
            blink_detected, new_symbol, blink_ended = self.vision_processor.detect_blink(
                avg_ear_s, symbol, current_time)
            
            if new_symbol:
                # User blinked during the correct symbol phase!
                # Add this symbol to our Morse sequence
                temp_seq = self.morse_sequence + new_symbol
                
                # Check if this could be part of a valid Morse code
                if is_valid_prefix(temp_seq):
                    self.morse_sequence = temp_seq
                    self.invalid_sequence_detected = False
                else:
                    # This sequence can't become any valid Morse code
                    self.invalid_sequence_detected = True
                    message = f"ERROR: '{temp_seq}' invalid prefix."
                    self.ui_handler.show_notification(message, current_time, is_error=True)
    
    def run(self):
        """Main program loop - this is where everything happens!"""
        print("Blink-to-Morse detector initializing...")
        print(f"Calibrating for {CALIBRATION_FRAMES} frames. Please keep your eyes open.")
        
        # Set up all our systems
        self.initialize_camera()
        self.audio_manager.start_audio_threads()
        
        try:
            running = True
            # Main loop: keep running until user quits or camera fails
            while running and self.cap.isOpened():
                # Get the next frame from camera
                success, frame = self.cap.read()
                if not success:
                    break
                
                # Mirror the image so it feels more natural
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()  # We'll draw on this copy
                current_time = time.time()    # Get current time for timing
                
                # Process the frame to detect face and calculate eye openness
                results, left_ear_s, right_ear_s, avg_ear_s = self.vision_processor.process_frame(frame)
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                running = self.handle_key_input(key, current_time)
                if not running:
                    break  # User pressed 'Q' to quit
                
                # CALIBRATION PHASE: Learn user's normal eye openness
                if not self.calibration_done:
                    self.frame_count += 1
                    status_text = f"CALIBRATING: {self.frame_count}/{CALIBRATION_FRAMES}. Keep eyes open."
                    
                    # Check if calibration is complete
                    if self.frame_count >= CALIBRATION_FRAMES:
                        self.vision_processor.complete_calibration()
                        self.calibration_done = True
                        self.sequence_start_time = current_time
                        status_text = f"CALIBRATION COMPLETE. Thresh: {self.vision_processor.EAR_CLOSED_THRESHOLD:.2f}"
                        print(status_text)
                        print("Sequential Mode: DOT(3s)->1s GAP->DASH(3s)->1s GAP->SPACE(3s)->1s GAP. All with Both Eyes Blink.")
                
                else:
                    # NORMAL OPERATION: User is entering Morse code
                    status_text = ""
                    self.update_sequential_input(current_time)  # Cycle through symbols
                    self.process_blink_input(avg_ear_s, current_time)  # Detect blinks
                
                # DRAW VISUAL INTERFACE
                
                # Draw eye landmarks if face is detected
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    self.vision_processor.draw_eye_landmarks(display_frame, landmarks)
                
                # Show eye openness readings and status
                self.ui_handler.draw_ear_readings(display_frame, left_ear_s, right_ear_s, status_text)
                
                # Draw the sequential input interface (DOT/DASH/SPACE cycling)
                if self.calibration_done:
                    elapsed_time = current_time - self.sequence_start_time
                    duration = INTERVAL_DURATION if self.is_interval_phase else self.current_symbol_data["duration"]
                    time_left = max(0.0, duration - elapsed_time)
                    
                    self.ui_handler.draw_sequential_ui(
                        display_frame, self.is_interval_phase, 
                        self.current_symbol_data["symbol"], time_left)
                
                # Create the side panel with Morse sequence and decoded text
                text_panel = self.ui_handler.create_text_panel(
                    display_frame.shape[0], self.morse_sequence, 
                    self.decoded_text, self.invalid_sequence_detected)
                
                # Combine camera view with text panel and show to user
                self.ui_handler.display_combined_frame(display_frame, text_panel)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Always clean up, even if something went wrong
            self.cleanup()
    
    def cleanup(self):
        """Shut down everything properly"""
        if self.cap:
            self.cap.release()  # Release camera
        
        # Clean up all our managers
        self.audio_manager.cleanup()
        self.vision_processor.cleanup()
        self.ui_handler.cleanup()
        
        print("Program terminated cleanly.")

# This is the entry point - when you run the script, this happens:
if __name__ == "__main__":
    app = EyeBlinkMorseApp()  # Create the application
    app.run()                 # Start the main loop