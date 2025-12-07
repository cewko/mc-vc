import logging
import time
import threading
from typing import Dict, Callable, Optional

from core.audio.audio_processor import AudioProcessor
from core.input.hotkey_manager import HotkeyManager
from core.system.minecraft_detector import MinecraftDetector
from services.message_sender import MessageSender
from config.constants import STATUS_UPDATE_DELAY


class VoiceService:
    """Main service coordinating voice chat functionality."""
    
    def __init__(
            self, 
            hotkey_mappings: Dict[str, str],
            status_callback: Optional[Callable[[str], None]] = None,
            model_ready_callback: Optional[Callable[[bool], None]] = None,
            auto_send: bool = True
        ):
        self._logger = logging.getLogger(__name__)
        self._hotkey_mappings = hotkey_mappings
        self._status_callback = status_callback
        self._model_ready_callback = model_ready_callback
        self._auto_send = auto_send
        
        # Initialize components
        self._audio_processor = AudioProcessor(
            self._on_transcription_complete, 
            self._on_model_ready,
            status_callback
        )
        self._hotkey_manager = HotkeyManager(
            hotkey_mappings,
            self._on_hotkey_pressed,
            self._on_hotkey_released
        )
        self._minecraft_detector = MinecraftDetector()
        self._message_sender = MessageSender()
        
        # State management
        self._is_running = False
        self._current_prefix = ""
        
        self._update_status(f"Voice service initialized ({'auto-send' if auto_send else 'manual-send'})")
    
    def initialize_model(self) -> None:
        """Initialize the speech recognition model."""
        self._logger.info("Initializing speech recognition model")
        self._update_status("Loading speech recognition model...")
        self._audio_processor.initialize_model()
    
    def _on_model_ready(self, is_ready: bool) -> None:
        """Handle model ready callback."""
        if is_ready:
            send_mode = "auto-send" if self._auto_send else "manual-send"
            self._update_status(f"Model loaded successfully! Ready to start ({send_mode})")
        else:
            self._update_status("Failed to load speech recognition model")
            
        if self._model_ready_callback:
            self._model_ready_callback(is_ready)
    
    def start(self) -> None:
        """Start the voice chat service."""
        if self._is_running:
            self._logger.warning("Service already running")
            return
        
        if not self._audio_processor.is_model_ready:
            self._update_status("Cannot start: Speech recognition model not ready")
            return
            
        self._is_running = True
        self._hotkey_manager.start_monitoring()
        
        send_mode = "auto-send" if self._auto_send else "manual-send"
        self._update_status(f"Started ({send_mode}). Use configured hotkeys to record")
        self._logger.info(f"VC service started with hotkeys: {self._hotkey_mappings}")
    
    def stop(self) -> None:
        """Stop the voice chat service."""
        if not self._is_running:
            return
            
        self._is_running = False
        self._hotkey_manager.stop_monitoring()
        self._audio_processor.cleanup()
        
        self._update_status("Stopped")
        self._logger.info("VC service stopped")
    
    def set_auto_send(self, auto_send: bool) -> None:
        """Update auto-send setting."""
        self._auto_send = auto_send
        self._message_sender.set_auto_send(auto_send)
        
        send_mode = "auto-send" if auto_send else "manual-send"
        self._logger.info(f"Auto-send mode changed to: {send_mode}")
    
    def update_hotkey_mappings(self, new_mappings: Dict[str, str]) -> None:
        """Update hotkey mappings."""
        self._hotkey_mappings = new_mappings
        self._hotkey_manager.update_hotkey_mappings(new_mappings)
    
    def _on_hotkey_pressed(self, prefix: str) -> None:
        """Handle hotkey press event."""
        if self._audio_processor.is_recording:
            return
        
        if not self._audio_processor.is_model_ready:
            self._update_status("Speech recognition model not ready")
            return
            
        if not self._minecraft_detector.is_minecraft_focused():
            self._update_status("Minecraft is not focused. Hold the key while in-game.")
            return
            
        self._current_prefix = prefix
        try:
            self._audio_processor.start_recording()
            send_mode = "auto-send" if self._auto_send else "manual-send"
            self._update_status(f"Recording with prefix '{prefix}' ({send_mode})...")
        except Exception as error:
            self._logger.error(f"Failed to start recording: {error}")
            self._update_status(f"Recording error")
    
    def _on_hotkey_released(self, prefix: str) -> None:
        """Handle hotkey release event."""
        if self._audio_processor.is_recording and self._current_prefix == prefix:
            self._audio_processor.stop_recording_and_process()
            self._update_status("Processing...")
    
    def _on_transcription_complete(self, transcribed_text: str) -> None:
        """Handle completed transcription."""
        if not transcribed_text:
            self._update_status("No speech detected")
            self._schedule_status_reset()
            return
            
        # Format message with prefix
        message = self._format_message(transcribed_text, self._current_prefix)
        
        try:
            if self._auto_send:
                self._update_status(f"Message recognized. Sending automatically")
                self._message_sender.send_message(message, auto_send=True)
            else:
                self._update_status(f"Message recognized. Ready to be modified (press Enter to send)")
                self._message_sender.send_message(message, auto_send=False)
                
        except Exception as error:
            self._logger.error(f"Failed to send message: {error}")
            self._update_status(f"Send error: {error}")
        
        self._schedule_status_reset()
    
    def _format_message(self, text: str, prefix: str) -> str:
        """Format message with prefix."""
        if prefix:
            return f"{prefix} {text}"
        return text
    
    def _schedule_status_reset(self) -> None:
        """Schedule status reset after delay."""
        def reset_status():
            time.sleep(STATUS_UPDATE_DELAY)
            send_mode = "auto-send" if self._auto_send else "manual-send"
            self._update_status(f"Use hotkeys to speak ({send_mode})")
        
        threading.Thread(target=reset_status, daemon=True, name="StatusReset").start()
    
    def _update_status(self, message: str) -> None:
        """Update status through callback."""
        if self._status_callback:
            self._status_callback(message)
    
    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._is_running
    
    @property
    def is_model_ready(self) -> bool:
        """Check if the speech recognition model is ready."""
        return self._audio_processor.is_model_ready