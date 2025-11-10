"""
Configuration settings for the Eye Blink Morse Code system
"""

# Camera settings
CAMERA_WIDTH = 1000
CAMERA_HEIGHT = 700
camera_index = 0
TARGET_FPS = 60

# Calibration settings
CALIBRATION_FRAMES = 150

# Eye Aspect Ratio (EAR) thresholds
EAR_OPEN_FACTOR = 0.65
EAR_HYSTERESIS_FACTOR = 0.95

# Noise reduction filter settings
MIN_FRAMES_TO_BLINK = 3
MIN_FRAMES_TO_OPEN = 3

# Blink timing parameters
MIN_BLINK_DURATION = 0.01
MAX_BLINK_DURATION = 1.50

# Display settings
TEXT_PANEL_WIDTH = 450
NOTIFICATION_DURATION = 2.0

# EAR smoothing window
EAR_SMOOTHING = 5

# Sequential input configuration
INPUT_SEQUENCE = [
    {"symbol": ".", "duration": 3.0, "text": "DOT (Both Eyes Blink)"},
    {"symbol": "-", "duration": 3.0, "text": "DASH (Both Eyes Blink)"},
    {"symbol": " ", "duration": 2.0, "text": "SPACE (Both Eyes Blink)"}
]

INTERVAL_DURATION = 1.0  # gap between symbols

# Morse code dictionary
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

# Audio settings
DOT_DURATION_MS = 200
DASH_DURATION_MS = 600
INTER_ELEMENT_GAP_MS = 200   # gap between dot/dash
INTER_LETTER_GAP_MS = 400    # gap between letters
INTER_WORD_GAP_MS = 800      # gap between words
BEEP_FREQUENCY = 750         # Hz

# Key handling
KEY_COOLDOWN = 0.3  # 300ms cooldown between key presses

# Eye landmark indices for MediaPipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]