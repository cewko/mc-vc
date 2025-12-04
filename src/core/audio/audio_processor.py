import logging
import threading
from typing import Callable, Optional

from core.audio.voice_recorder import VoiceRecorder
from core.audio.speech_recognizer import SpeechRecognizer
from utils.exceptions import AudioProcessingError


class AudioProcessor:
    """Coordinates audio recording and speech recognition."""
    
    def __init__(
        self, 
        transcription_callback: Callable[[str], None],
        model_ready_callback: Optional[Callable[[bool], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ):
        self._logger = logging.getLogger(__name__)
        self._transcription_callback = transcription_callback
        self._model_ready_callback = model_ready_callback
        self._status_callback = status_callback
        
        self._voice_recorder = VoiceRecorder(buffer_full_callback=self._on_buffer_full)
        self._speech_recognizer = SpeechRecognizer(self._on_model_ready)
        
    def initialize_model(self) -> None:
        """Initialize the speech recognition model asynchronously."""
        self._logger.info("Initializing speech recognition model")
        self._speech_recognizer.initialize_model_async()

    def _on_buffer_full(self, message: str) -> None:
        self._logger.warning(message)
        if self._status_callback:
            self._status_callback(message)
    
    def _on_model_ready(self, is_ready: bool) -> None:
        """Handle model ready callback."""
        if is_ready:
            self._logger.info("Speech recognition model is ready")
        else:
            self._logger.error("Speech recognition model failed to load")
            
        if self._model_ready_callback:
            self._model_ready_callback(is_ready)
        
    def start_recording(self) -> None:
        """Start audio recording."""
        if not self._speech_recognizer.is_model_ready:
            raise AudioProcessingError("Speech recognition model is not ready")
            
        try:
            self._voice_recorder.start_recording()
            self._logger.info("Audio recording started")
        except Exception as error:
            raise AudioProcessingError(f"Failed to start recording: {error}")
    
    def stop_recording_and_process(self) -> None:
        """Stop recording and process audio in a separate thread."""
        if not self._voice_recorder.is_recording:
            self._logger.warning("No recording in progress")
            return
            
        try:
            audio_data = self._voice_recorder.stop_recording()
            processing_thread = threading.Thread(
                target=self._process_audio,
                args=(audio_data,),
                daemon=True,
                name="AudioProcessor"
            )
            processing_thread.start()
            
        except AudioProcessingError as error:
            self._logger.error(f"Recording failed: {error}")
            self._transcription_callback(None)
    
    def _process_audio(self, audio_data) -> None:
        """Process audio data and call transcription callback."""
        try:
            transcribed_text = self._speech_recognizer.transcribe(audio_data)
            self._transcription_callback(transcribed_text)
            
        except AudioProcessingError as error:
            self._logger.error(f"Audio processing failed: {error}")
            self._transcription_callback(None)
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._voice_recorder.is_recording
    
    @property
    def is_model_ready(self) -> bool:
        """Check if the speech recognition model is ready."""
        return self._speech_recognizer.is_model_ready
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._voice_recorder.cleanup()