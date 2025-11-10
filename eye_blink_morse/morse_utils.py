"""
Morse code utility functions and conversions
"""
from config import MORSE_DICT

# Precompute valid sequences and reverse mapping
VALID_MORSE_SEQUENCES = set(MORSE_DICT.keys())
LETTER_TO_MORSE = {v: k for k, v in MORSE_DICT.items()}
LETTER_TO_MORSE[' '] = ' '  # keep space

def morse_to_letter(morse_seq: str) -> str:
    """
    Convert Morse code sequence to corresponding letter/number
    
    Args:
        morse_seq: Morse code string (e.g., ".-")
    
    Returns:
        Corresponding character or empty string if not found
    """
    return MORSE_DICT.get(morse_seq, "")

def is_valid_prefix(morse_seq: str) -> bool:
    """
    Check if a Morse sequence is a valid prefix of any known code
    
    Args:
        morse_seq: Morse code string to check
    
    Returns:
        True if valid prefix, False otherwise
    """
    if not morse_seq:
        return True
    for code in VALID_MORSE_SEQUENCES:
        if code.startswith(morse_seq):
            return True
    return False

def text_to_morse(text: str) -> str:
    """
    Convert text into Morse code string with proper spacing
    
    Args:
        text: Input text to convert
    
    Returns:
        Morse code string with single spaces between letters and double spaces between words
    """
    if not text:
        return ""
    
    parts = []
    words = text.split(" ")
    
    for word in words:
        letters = []
        for ch in word:
            mor = LETTER_TO_MORSE.get(ch.upper(), "")
            if mor:
                letters.append(mor)
        parts.append(" ".join(letters))
    
    # Join words with double space
    return "  ".join(parts)