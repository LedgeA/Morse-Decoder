import cv2
import mediapipe as mp
import time
import math
import numpy as np
from collections import deque

# -------------------- CONFIG --------------------
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 750
camera_index = 0          
TARGET_FPS = 60            

# Dynamic thresholding parameters
CALIBRATION_FRAMES = 150

# Using Eye Aspect Ratio (EAR)
EAR_OPEN_FACTOR = 0.65
EAR_HYSTERESIS_FACTOR = 0.95

# Noise Reduction Filter
MIN_FRAMES_TO_BLINK = 3
MIN_FRAMES_TO_OPEN = 3

# Timing parameters
MIN_BLINK_DURATION = 0.01
MAX_BLINK_DURATION = 1.50

# Display
TEXT_PANEL_WIDTH = 450
NOTIFICATION_DURATION = 2.0

# EAR smoothing window
EAR_SMOOTHING = 5

# --- NEW SEQUENTIAL INPUT CONFIG (with intervals) ---
INPUT_SEQUENCE = [
    {"symbol": ".", "duration": 3.0, "text": "DOT (Both Eyes)"},
    {"symbol": "-", "duration": 3.0, "text": "DASH (Left Eye)"},
    {"symbol": " ", "duration": 1.0, "text": "SPACE (Right Eye)"}
]

INTERVAL_DURATION = 1.0  # gap between symbols

# -------------------------------------------------
# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Eye landmark indices (P1, P2, P3, P4, P5, P6)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Morse dictionary
MORSE_DICT = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9"
}
VALID_MORSE_SEQUENCES = set(MORSE_DICT.keys())


# -------------------- FUNCTIONS --------------------
def euclidean_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def eye_aspect_ratio(landmarks, eye_indices):
    A = euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    B = euclidean_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    C = euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def morse_to_letter(morse_seq):
    return MORSE_DICT.get(morse_seq, "")


def is_valid_prefix(morse_seq):
    if not morse_seq:
        return True
    for code in VALID_MORSE_SEQUENCES:
        if code.startswith(morse_seq):
            return True
    return False


# -------------------- INITIALIZE --------------------
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Attempted to set FPS to {TARGET_FPS}. Actual FPS: {actual_fps}")

window_title = "Eye Blink Morse Code (Sequential Input with Interval)"
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_title, 1600, 900)

blink_start_time = None
blinking = False
morse_sequence = ""
decoded_text = ""
EAR_CLOSED_THRESHOLD = None
EAR_OPEN_THRESHOLD = None
open_ear_values = []
calibration_done = False
invalid_sequence_detected = False
blink_type = None
notification_message = ""
notification_time = 0.0
blink_frame_counter = 0
open_frame_counter = 0

# Sequential input state
sequence_index = 0
sequence_start_time = 0.0
current_symbol_data = INPUT_SEQUENCE[0]
is_interval_phase = False

# smoothing histories
left_ear_hist = deque(maxlen=EAR_SMOOTHING)
right_ear_hist = deque(maxlen=EAR_SMOOTHING)
avg_ear_hist = deque(maxlen=EAR_SMOOTHING)

print("Blink-to-Morse detector initializing...")
print(f"Calibrating for {CALIBRATION_FRAMES} frames. Please keep your eyes open.")

# -------------------- MAIN LOOP --------------------
frame_count = 0
status_text = "Waiting for face..."
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    current_time = time.time()
    display_frame = frame.copy()
    h, w, _ = display_frame.shape

    key = cv2.waitKey(1) & 0xFF

    if key in [ord('r'), ord('R')]:
        morse_sequence = ""
        invalid_sequence_detected = False
        notification_message = "RESET: Current Morse sequence cleared."
        notification_time = current_time
        print(notification_message)

    elif key == 13:  # Enter (submit morse code to decode)
        if morse_sequence:
            letter = morse_to_letter(morse_sequence)
            if letter:
                decoded_text += letter
                notification_message = f"SUBMITTED: '{morse_sequence}' decoded to '{letter}'"
                print(notification_message)
            else:
                notification_message = f"ERROR: '{morse_sequence}' is not valid."
                print(notification_message)
            morse_sequence = ""
            invalid_sequence_detected = False
            notification_time = current_time
        else:
            notification_message = "ERROR: No Morse sequence to submit."
            notification_time = current_time

    elif key in [ord('d'), ord('D')]:
        if decoded_text:
            deleted_char = decoded_text[-1]
            decoded_text = decoded_text[:-1]
            notification_message = f"DELETED: Removed '{deleted_char}'."
            notification_time = current_time
            print(notification_message)
        else:
            notification_message = "ERROR: Nothing to delete."
            notification_time = current_time

    elif key in [ord('q'), ord('Q')]:
        break

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        left_ear_hist.append(left_ear)
        right_ear_hist.append(right_ear)
        avg_ear_hist.append(avg_ear)

        left_ear_s = sum(left_ear_hist) / len(left_ear_hist)
        right_ear_s = sum(right_ear_hist) / len(right_ear_hist)
        avg_ear_s = sum(avg_ear_hist) / len(avg_ear_hist)

        for eye_indices in [LEFT_EYE, RIGHT_EYE]:
            points = []
            for idx in eye_indices:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                points.append((x, y))
                cv2.circle(display_frame, (x, y), 2, (0, 255, 0), -1)
            for i in range(len(points)):
                cv2.line(display_frame, points[i], points[(i + 1) % len(points)], (0, 255, 0), 1)

        cv2.putText(display_frame, f"L-EAR: {left_ear_s:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_frame, f"R-EAR: {right_ear_s:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if not calibration_done:
            if frame_count < CALIBRATION_FRAMES:
                open_ear_values.append(avg_ear_s)
                frame_count += 1
                status_text = f"CALIBRATING: {frame_count}/{CALIBRATION_FRAMES}. Keep eyes open."
            else:
                baseline_open_ear = float(np.percentile(open_ear_values, 80))
                EAR_CLOSED_THRESHOLD = baseline_open_ear * EAR_OPEN_FACTOR
                EAR_OPEN_THRESHOLD = EAR_CLOSED_THRESHOLD * EAR_HYSTERESIS_FACTOR
                calibration_done = True
                sequence_start_time = current_time
                status_text = f"CALIBRATION COMPLETE. Thresh: {EAR_CLOSED_THRESHOLD:.2f}"
                print(status_text)
                print("Sequential Mode: DOT(3s)->1s GAP->DASH(3s)->1s GAP->SPACE(1s)->1s GAP")
            cv2.putText(display_frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if calibration_done:
            elapsed_time = current_time - sequence_start_time

            # --- Handle symbol / interval switching ---
            duration = INTERVAL_DURATION if is_interval_phase else current_symbol_data["duration"]
            if elapsed_time > duration:
                if is_interval_phase:
                    sequence_index = (sequence_index + 1) % len(INPUT_SEQUENCE)
                    current_symbol_data = INPUT_SEQUENCE[sequence_index]
                    is_interval_phase = False
                else:
                    is_interval_phase = True
                sequence_start_time = current_time
                elapsed_time = 0.0

            if is_interval_phase:
                cv2.putText(display_frame, "INTERVAL - REST", (w // 2 - 200, h - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 3)
            else:
                symbol = current_symbol_data["symbol"]
                icon_text = "DOT" if symbol == "." else "DASH" if symbol == "-" else "SPACE"
                color = (0, 255, 0) if symbol == "." else (255, 255, 0) if symbol == "-" else (255, 0, 0)

                cv2.putText(display_frame, f"ACTIVE: {icon_text}", (w // 2 - 150, h - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                cv2.putText(display_frame, f"TIME LEFT: {max(0.0, duration - elapsed_time):.1f}s",
                            (w // 2 - 150, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                # ---- Blink Detection only when active ----
                is_left_closed_raw = left_ear_s < EAR_CLOSED_THRESHOLD
                is_right_closed_raw = right_ear_s < EAR_CLOSED_THRESHOLD
                is_left_open_raw = left_ear_s > EAR_OPEN_THRESHOLD
                is_right_open_raw = right_ear_s > EAR_OPEN_THRESHOLD

                potential_blink_type = None
                if is_left_closed_raw and is_right_closed_raw:
                    potential_blink_type = '.'
                elif is_left_closed_raw and is_right_open_raw:
                    potential_blink_type = '-'
                elif is_right_closed_raw and is_left_open_raw:
                    potential_blink_type = ' '

                if not blinking:
                    if potential_blink_type == symbol:
                        blink_frame_counter += 1
                        open_frame_counter = 0
                        if blink_frame_counter >= MIN_FRAMES_TO_BLINK:
                            blinking = True
                            blink_start_time = current_time
                            blink_type = potential_blink_type
                            cv2.putText(display_frame, f"INPUT DETECTED: {icon_text}", (w // 2 - 100, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    else:
                        blink_frame_counter = 0

                if blinking:
                    if is_left_open_raw and is_right_open_raw:
                        open_frame_counter += 1
                        blink_frame_counter = 0
                        if open_frame_counter >= MIN_FRAMES_TO_OPEN:
                            blink_duration = current_time - blink_start_time
                            new_symbol = None
                            if MIN_BLINK_DURATION <= blink_duration <= MAX_BLINK_DURATION:
                                if blink_type == '.':
                                    new_symbol = "."
                                elif blink_type == '-':
                                    new_symbol = "-"
                                elif blink_type == ' ':
                                    if decoded_text and decoded_text[-1] != " ":
                                        decoded_text += " "
                                        notification_message = "SPACE added."
                                        notification_time = current_time
                            if new_symbol:
                                temp_seq = morse_sequence + new_symbol
                                if is_valid_prefix(temp_seq):
                                    morse_sequence = temp_seq
                                else:
                                    invalid_sequence_detected = True
                                    notification_message = f"ERROR: '{temp_seq}' invalid prefix."
                                    notification_time = current_time
                            blinking = False
                            blink_start_time = None
                            blink_type = None
                            open_frame_counter = 0
                    elif blink_start_time and (current_time - blink_start_time > MAX_BLINK_DURATION):
                        blinking = False
                        blink_start_time = None
                        blink_type = None
                        blink_frame_counter = 0
                        open_frame_counter = 0

    else:
        cv2.putText(display_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # TEXT PANEL
    text_panel = np.ones((h, TEXT_PANEL_WIDTH, 3), dtype=np.uint8) * 255
    cv2.putText(text_panel, "MORSE INPUT (Sequential)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.line(text_panel, (10, 40), (TEXT_PANEL_WIDTH - 10, 40), (200, 200, 200), 1)
    cv2.putText(text_panel, "Current Sequence:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    color = (255, 0, 0) if not invalid_sequence_detected else (0, 0, 255)
    cv2.putText(text_panel, morse_sequence, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
    cv2.putText(text_panel, "DECODED TEXT", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.line(text_panel, (10, 190), (TEXT_PANEL_WIDTH - 10, 190), (200, 200, 200), 1)

    lines, line = [], ""
    for word in decoded_text.split(" "):
        if len(line) + len(word) + 1 > 30:
            lines.append(line)
            line = word
        else:
            line += (" " if line else "") + word
    if line:
        lines.append(line)
    y_offset = 230
    for l in lines:
        cv2.putText(text_panel, l, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        y_offset += 35
        if y_offset > h - 100:
            break
    cv2.putText(text_panel, "SUBMIT [ENTER] | RESET [R] | DELETE [D]", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
    if notification_message and (current_time - notification_time < NOTIFICATION_DURATION):
        color = (0, 0, 255) if "ERROR" in notification_message else (255, 0, 0)
        cv2.putText(text_panel, notification_message, (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    combined = cv2.hconcat([display_frame, text_panel])
    cv2.imshow(window_title, combined)

cap.release()
cv2.destroyAllWindows()
