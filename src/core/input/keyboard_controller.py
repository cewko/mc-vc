import logging
import time
import pyperclip
from pynput.keyboard import Key, Controller

from config.constants import MINECRAFT_CHAT_KEY, ENTER_KEY_DELAY
from utils.exceptions import MessageSendError


class KeyboardController:
    """Handles keyboard input simulation."""
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._controller = Controller()
    
    def send_message_to_minecraft(self, message: str, auto_send: bool = True) -> None:
        """Send a message to Minecraft chat using clipboard paste."""
        try:
            original_clipboard = None
            try:
                original_clipboard = pyperclip.paste()
            except:
                # ignore if clipboard is empty or inaccesible
                pass
            
            pyperclip.copy(message)
            
            self.simulate_key_press(MINECRAFT_CHAT_KEY)
            
            self._controller.press(Key.ctrl)
            time.sleep(0.05)
            self._controller.press('v')
            time.sleep(0.05)
            self._controller.release('v')
            time.sleep(0.05)
            self._controller.release(Key.ctrl)
            
            if auto_send:
                self.simulate_key_press(Key.enter)
                self._logger.info(f"Sent to Minecraft chat: '{message}'")
            else:
                self._logger.info(f"Typed in Minecraft chat: '{message}' (manual send)")
            
            # restore clipboard to prev state after everything is done
            if original_clipboard is not None:
                time.sleep(0.15)
                try:
                    pyperclip.copy(original_clipboard)
                except:
                    pass
                
        except Exception as error:
            raise MessageSendError(f"Failed to send message to Minecraft: {error}")
    
    def simulate_key_press(self, key: str) -> None:
        """Simulate a key press."""
        try:
            self._controller.press(key)
            self._controller.release(key)
        except Exception as error:
            self._logger.error(f"Failed to simulate key press for '{key}': {error}")

