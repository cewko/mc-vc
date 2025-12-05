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
        self._stream_active = False 
        self._buffer_lock = Lock()
        
    def start_recording(self) -> None:
        """Start audio recording."""
        if self._is_recording:
            self._logger.warning("Recording already in progress")
            return
        
        if self._stream_active:
            self._cleanup_stream()
            
        try:
            # Reset buffer position
            self._buffer_position = 0
            self._is_recording = True
            self._stream_active = True
            
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
            self._stream_active = False
            self._cleanup_stream()
            raise AudioProcessingError(f"Failed to start recording: {e}")
    
    def stop_recording(self) -> np.ndarray:
        """Stop audio recording and return recorded audio."""
        was_recording = self._is_recording
        self._is_recording = False
        
        if not was_recording and not self._stream_active:
            self._logger.warning("No recording in progress")
            return np.array([], dtype=np.float32)
        
        try:
            self._cleanup_stream()
                
            with self._buffer_lock:
                if self._buffer_position == 0:
                    raise AudioProcessingError("No audio data captured")
                
                # Return only the recorded portion
                audio_data = self._audio_buffer[:self._buffer_position].copy()
                
            self._logger.info(f"Recording stopped: {self._buffer_position} samples captured")
            return audio_data
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to stop recording: {e}")
    
    def _cleanup_stream(self) -> None:
        """Safely cleanup the audio stream."""
        self._stream_active = False
        if self._stream is not None:
            try:
                if self._stream.active:
                    self._stream.stop()
                self._stream.close()
            except Exception as error:
                self._logger.warning(f"Error cleaning up stream: {error}")
            finally:
                self._stream = None
    
    def _audio_callback(self, indata: np.ndarray, frames, _time_info, status) -> None:
        """Audio stream callback function."""
        if not self._is_recording:
            return
            
        with self._buffer_lock:
            chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            chunk_size = len(chunk)
            
            # Check if we have space
            if self._buffer_position + chunk_size > self._max_samples:
                self._logger.warning(f"Recording buffer full ({self._max_recording_duration}s limit reached)")
                self._is_recording = False  # stop accepting data but stream cleanup happens in stop_recording
                
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
        """Check if currently recording or stream needs cleanup."""
        return self._is_recording or self._stream_active
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._is_recording = False
        self._cleanup_stream()