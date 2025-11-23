from config import MORSE_DICT

# Precompute valid sequences and reverse mapping
VALID_MORSE_SEQUENCES = set(MORSE_DICT.keys())
LETTER_TO_MORSE = {v: k for k, v in MORSE_DICT.items()}
LETTER_TO_MORSE[' '] = ' '  # keep space

def morse_to_letter(morse_seq: str) -> str:
    return MORSE_DICT.get(morse_seq, "")

def is_valid_prefix(morse_seq: str) -> bool:
    if not morse_seq:
        return True
    for code in VALID_MORSE_SEQUENCES:
        if code.startswith(morse_seq):
            return True
    return False

def text_to_morse(text: str) -> str:
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