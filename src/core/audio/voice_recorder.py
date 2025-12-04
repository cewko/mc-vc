import logging
import sounddevice as sd
import numpy as np
from typing import Optional, Callable
from threading import Lock

from config.settings import app_settings
from utils.exceptions import AudioProcessingError


class VoiceRecorder:
    """Handles audio recording and buffering."""
    
    def __init__(self, buffer_full_callback: Optional[Callable[[str], None]] = None):
        self._logger = logging.getLogger(__name__)
        self._audio_settings = app_settings.audio
        self._buffer_full_callback = buffer_full_callback
        
        # Pre-allocated buffer optimization
        self._max_recording_duration = 20
        self._max_samples = self._audio_settings.sample_rate * self._max_recording_duration
        self._audio_buffer = np.zeros(self._max_samples, dtype=np.float32)
        self._buffer_position = 0
        
        self._stream: Optional[sd.InputStream] = None
        self._is_recording = False
        self._buffer_lock = Lock()
        
    def start_recording(self) -> None:
        """Start audio recording."""
        if self._is_recording:
            self._logger.warning("Recording already in progress")
            return
            
        try:
            # Reset buffer position
            self._buffer_position = 0
            self._is_recording = True
            
            self._stream = sd.InputStream(
                samplerate=self._audio_settings.sample_rate,
                channels=self._audio_settings.channels,
                dtype=self._audio_settings.dtype,
                callback=self._audio_callback
            )
            self._stream.start()
            self._logger.info("Recording started")
            
        except Exception as e:
            self._is_recording = False
            raise AudioProcessingError(f"Failed to start recording: {e}")
    
    def stop_recording(self) -> np.ndarray:
        """Stop audio recording and return recorded audio."""
        if not self._is_recording:
            self._logger.warning("No recording in progress")
            return np.array([])
            
        self._is_recording = False
        
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
                
            with self._buffer_lock:
                if self._buffer_position == 0:
                    raise AudioProcessingError("No audio data captured")
                
                # Return only the recorded portion (not the entire 30-second buffer)
                audio_data = self._audio_buffer[:self._buffer_position].copy()
                
            self._logger.info(f"Recording stopped: {self._buffer_position} samples captured")
            return audio_data
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to stop recording: {e}")
    
    def _audio_callback(self, indata: np.ndarray, frames, _time_info, _status) -> None:
        """Audio stream callback function."""
        if not self._is_recording:
            return
            
        with self._buffer_lock:
            # Flatten to 1D if stereo (though we're using mono)
            chunk = indata.flatten() if indata.ndim > 1 else indata[:, 0]
            chunk_size = len(chunk)
            
            # Check if we have space
            if self._buffer_position + chunk_size > self._max_samples:
                self._logger.warning(f"Recording buffer full ({self._max_recording_duration}s limit reached)")
                self._is_recording = False
                
                if self._buffer_full_callback:
                    self._buffer_full_callback(
                        f"Recording stopped: {self._max_recording_duration}s buffer limit reached"
                    )
                    return
            
            # Copy directly into pre-allocated buffer
            self._audio_buffer[self._buffer_position:self._buffer_position + chunk_size] = chunk
            self._buffer_position += chunk_size
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._is_recording:
            self.stop_recording()