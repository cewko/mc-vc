import logging
import numpy as np
import time
import threading
from typing import Optional, Callable
from faster_whisper import WhisperModel

from config.settings import app_settings
from config.constants import DEFAULT_SENTENCE_ENDINGS
from utils.exceptions import AudioProcessingError


class SpeechRecognizer:
    """Handles speech-to-text conversion using Whisper."""
    
    def __init__(self, model_ready_callback: Optional[Callable[[bool], None]] = None):
        self._logger = logging.getLogger(__name__)
        self._audio_settings = app_settings.audio
        self._model: Optional[WhisperModel] = None
        self._model_dir = self._audio_settings.get_models_path()
        self._model_ready_callback = model_ready_callback
        self._is_model_loading = False
        self._is_model_ready = False
        self._model_load_lock = threading.Lock()
        
    def initialize_model_async(self) -> None:
        """Initialize the Whisper model asynchronously."""
        if self._is_model_loading or self._is_model_ready:
            return
            
        self._is_model_loading = True
        self._logger.info("Starting asynchronous model initialization")
        
        initialization_thread = threading.Thread(
            target=self._load_model_sync,
            daemon=True,
            name="ModelInitializer"
        )
        initialization_thread.start()
    
    def _load_model_sync(self) -> None:
        """Synchronously load the Whisper model."""
        try:
            with self._model_load_lock:
                if self._model is not None:
                    return
                    
                self._logger.info(f"Loading Whisper model '{self._audio_settings.whisper_model}' from {self._model_dir}")
                
                self._model = WhisperModel(
                    self._audio_settings.whisper_model,
                    compute_type=self._audio_settings.compute_type,
                    download_root=str(self._model_dir)
                )

                self._is_model_ready = True
                self._logger.info(f"Whisper model '{self._audio_settings.whisper_model}' loaded successfully")
                
        except Exception as e:
            self._logger.error(f"Failed to load Whisper model: {e}")
            self._is_model_ready = False
            
        finally:
            self._is_model_loading = False
            if self._model_ready_callback:
                self._model_ready_callback(self._is_model_ready)
    
    def wait_for_model(self, timeout: Optional[float] = None) -> bool:
        """Wait for the model to be ready."""
        if self._is_model_ready:
            return True
            
        if not self._is_model_loading:
            self._logger.warning("Model is not loading. Call initialize_model_async() first.")
            return False
        
        start_time = time.time()
        poll_interval = 0.1
        
        while self._is_model_loading:
            if timeout and (time.time() - start_time) > timeout:
                self._logger.warning(f"Model loading timeout after {timeout} seconds")
                return False
                
            time.sleep(poll_interval)
        
        return self._is_model_ready
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text."""
        if audio_data.size == 0:
            raise AudioProcessingError("No audio data provided")
        
        if not self._is_model_ready:
            if self._is_model_loading:
                self._logger.info("Waiting for model to finish loading...")
                if not self.wait_for_model(timeout=30.0):
                    raise AudioProcessingError("Model loading timeout or failed")
            else:
                raise AudioProcessingError("Model not initialized. Call initialize_model_async() first.")
            
        try:
            if self._model is None:
                raise AudioProcessingError("Model is None despite being marked as ready")
            
            segments, _ = self._model.transcribe(
                audio_data, 
                language='en',
                vad_filter=True, 
                vad_parameters=dict(min_silence_duration_ms=500),
                beam_size=self._audio_settings.beam_size,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False
            )
                
            text = ''.join([segment.text for segment in segments]).strip()
            
            if not text:
                return ""
                
            formatted_text = self._format_text(text)
            self._logger.info(f"Transcribed text: '{formatted_text}'")
            
            return formatted_text
            
        except Exception as e:
            raise AudioProcessingError(f"Transcription failed: {e}")
    
    def _format_text(self, text: str) -> str:
        """Format transcribed text with proper capitalization and punctuation."""
        if not text:
            return text
            
        # Capitalize first letter
        formatted = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Add punctuation if missing
        if not formatted.endswith(DEFAULT_SENTENCE_ENDINGS):
            formatted += "."
            
        return formatted
    
    @property
    def is_model_ready(self) -> bool:
        """Check if the model is ready for transcription."""
        return self._is_model_ready