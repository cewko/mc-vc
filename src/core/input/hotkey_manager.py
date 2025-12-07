import logging
from typing import Dict, Callable, Set
from threading import Lock

import keyboard

from utils.exceptions import HotkeyError


class HotkeyManager:
    """Manages hotkey detection and callbacks."""
    
    def __init__(
            self, hotkey_mappings: Dict[str, str],
            start_callback: Callable[[str], None],
            stop_callback: Callable[[str], None]
        ):

        self._logger = logging.getLogger(__name__)
        self._validate_hotkey_mappings(hotkey_mappings)
        
        self._hotkey_mappings = hotkey_mappings
        self._start_callback = start_callback
        self._stop_callback = stop_callback
        
        self._pressed_keys: Set[str] = set()
        self._is_running = False
        self._key_lock = Lock()
        self._hooks = []
        
    def _validate_hotkey_mappings(self, mappings: Dict[str, str]) -> None:
        """Validate hotkey mappings."""
        if not mappings:
            raise HotkeyError("Hotkey mappings cannot be empty")
            
        # Check for duplicate hotkeys
        hotkeys = list(mappings.keys())
        if len(hotkeys) != len(set(hotkeys)):
            raise HotkeyError("Duplicate hotkeys found in mappings")
            
        # Validate hotkey format
        for hotkey in hotkeys:
            if not hotkey or not isinstance(hotkey, str):
                raise HotkeyError(f"Invalid hotkey format: {hotkey}")
        
    def start_monitoring(self) -> None:
        """Start monitoring hotkeys."""
        self._is_running = True

        for hotkey, prefix in self._hotkey_mappings.items():
            press_hook = keyboard.on_press_key(
                hotkey.lower(),
                lambda e, p=prefix: self._on_key_down(p),
                suppress=False
            )
            release_hook = keyboard.on_release_key(
                hotkey.lower(),
                lambda e, p=prefix: self._on_key_up(p),
                suppress=False
            )
            self._hooks.extend([press_hook, release_hook])

        self._logger.info(f"Started keyboard monitoring for: {list(self._hotkey_mappings.keys())}")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring hotkeys and cleanup"""
        if not self._is_running:
            return
        
        self._is_running = False

        for hook in self._hooks:
            keyboard.unhook(hook)

        self._hooks.clear()

        with self._key_lock:
            self._pressed_keys.clear()

        self._logger.info("Stopped hotkey monitoring")

    def _on_key_down(self, prefix: str) -> None:
        if not self._is_running:
            return
        
        with self._key_lock:
            if prefix not in self._pressed_keys:
                self._pressed_keys.add(prefix)
                self._start_callback(prefix)

    def _on_key_up(self, prefix: str) -> None:
        if not self._is_running:
            return
        
        with self._key_lock:
            if prefix in self._pressed_keys:
                self._pressed_keys.remove(prefix)
                self._stop_callback(prefix)
    
    def update_hotkey_mappings(self, new_mappings: Dict[str, str]) -> None:
        """Update hotkey mappings."""
        self._validate_hotkey_mappings(new_mappings)
        
        with self._key_lock:
            self._hotkey_mappings = new_mappings.copy()
            self._pressed_keys.clear()
        
        self._logger.info(f"Updated hotkey mappings: {new_mappings}")

