"""
Audio utilities for TTS and Morse code beeps
"""
import time
import threading
import pyttsx3
from queue import Queue
import sys
from config import *

# Try to import winsound (Windows)
try:
    import winsound
    WINSOUND_AVAILABLE = True
except Exception:
    WINSOUND_AVAILABLE = False
    print("Warning: winsound not available on this platform. Beeps will be disabled.", file=sys.stderr)

class AudioManager:
    """
    Manages text-to-speech and Morse code audio playback
    """
    
    def __init__(self):
        """Initialize audio components"""
        self._init_tts()
        self._init_beep_system()
        
    def _init_tts(self):
        """Initialize text-to-speech engine"""
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 100)
        self.tts_engine.setProperty('volume', 1.0)
        
        # Thread-safe queue for TTS
        self.speech_queue = Queue()
        self.tts_stop_event = threading.Event()
        self.tts_thread = None
        
    def _init_beep_system(self):
        """Initialize Morse code beep system"""
        self.beep_queue = Queue()
        self.beep_stop_event = threading.Event()
        self.beep_thread = None
    
    def start_audio_threads(self):
        """Start background audio threads"""
        self.tts_thread = self._start_tts_thread()
        self.beep_thread = self._start_beep_thread()
    
    def _tts_thread_worker(self):
        """Worker for the TTS thread (non-blocking speech)"""
        while not self.tts_stop_event.is_set():
            try:
                text = self.speech_queue.get(timeout=0.1)
                # Clear any previous speaking and speak new text
                try:
                    self.tts_engine.stop()
                except Exception:
                    pass
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                self.speech_queue.task_done()
            except Exception:
                pass
    
    def _beep_thread_worker(self):
        """Worker for playing beeps from the beep_queue (non-blocking)"""
        while not self.beep_stop_event.is_set():
            try:
                text_to_play = self.beep_queue.get(timeout=0.1)
                # Convert text to morse and play it
                from morse_utils import text_to_morse
                morse_string = text_to_morse(text_to_play)
                self._play_morse_blocking(morse_string)
                self.beep_queue.task_done()
            except Exception:
                pass
    
    def _start_tts_thread(self):
        """Start TTS thread"""
        thread = threading.Thread(target=self._tts_thread_worker, daemon=True)
        thread.start()
        return thread
    
    def _start_beep_thread(self):
        """Start beep thread"""
        thread = threading.Thread(target=self._beep_thread_worker, daemon=True)
        thread.start()
        return thread
    
    def _play_morse_blocking(self, morse_string: str):
        """
        Blocking playback function for Morse code
        
        Args:
            morse_string: Morse code string with dots, dashes, and spaces
        """
        if not morse_string:
            return

        # Iterate through morse string and play appropriate sounds
        i = 0
        L = len(morse_string)
        while i < L:
            ch = morse_string[i]
            if ch == '.':
                if WINSOUND_AVAILABLE:
                    winsound.Beep(BEEP_FREQUENCY, DOT_DURATION_MS)
                else:
                    time.sleep(DOT_DURATION_MS / 1000.0)
                time.sleep(INTER_ELEMENT_GAP_MS / 1000.0)
                i += 1
            elif ch == '-':
                if WINSOUND_AVAILABLE:
                    winsound.Beep(BEEP_FREQUENCY, DASH_DURATION_MS)
                else:
                    time.sleep(DASH_DURATION_MS / 1000.0)
                time.sleep(INTER_ELEMENT_GAP_MS / 1000.0)
                i += 1
            elif ch == ' ':
                # Count consecutive spaces
                j = i
                while j < L and morse_string[j] == ' ':
                    j += 1
                count = j - i
                if count == 1:
                    # Letter gap
                    time.sleep(INTER_LETTER_GAP_MS / 1000.0)
                else:
                    # Word gap (two or more spaces)
                    time.sleep(INTER_WORD_GAP_MS / 1000.0)
                i = j
            else:
                # Unknown character: skip
                i += 1
    
    def speak_text_non_blocking(self, text: str):
        """
        Queue text for non-blocking speaking
        
        Args:
            text: Text to speak
        """
        if not text:
            return
        # Clear queue and add new text
        with self.speech_queue.mutex:
            self.speech_queue.queue.clear()
        self.speech_queue.put(text)
    
    def play_text_as_morse_non_blocking(self, text: str):
        """
        Queue text to be played as Morse code (non-blocking)
        
        Args:
            text: Text to convert to Morse and play
        """
        if not text:
            return
        with self.beep_queue.mutex:
            self.beep_queue.queue.clear()
        self.beep_queue.put(text)
    
    def cleanup(self):
        """Clean up audio resources"""
        self.tts_stop_event.set()
        self.beep_stop_event.set()
        
        try:
            if self.tts_thread:
                self.tts_thread.join(timeout=1.0)
            if self.beep_thread:
                self.beep_thread.join(timeout=1.0)
        except Exception:
            pass
        
        try:
            self.tts_engine.stop()
        except Exception:
            pass