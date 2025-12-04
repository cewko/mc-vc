from dataclasses import dataclass
from typing import Dict
from pathlib import Path


@dataclass
class AudioSettings:
    """Audio processing configuration."""
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = 'float32'
    whisper_model: str = 'small'
    compute_type: str = 'int8'
    models_directory: str = None

    def __post_init__(self):
        if self.models_directory is None:
            self.models_directory = str(self._get_default_models_directory())

    def _get_default_models_directory(self) -> Path:
        """Get the default models directory path."""
        project_root = Path(__file__).parent.parent.parent
        return project_root / "models"

    def get_models_path(self) -> Path:
        """Get the models directory as a Path object with creation."""
        models_path = Path(self.models_directory)
        
        try:
            models_path.mkdir(parents=True, exist_ok=True)
            return models_path
        except (OSError, PermissionError):
            fallback_path = Path.cwd() / "models"
            fallback_path.mkdir(exist_ok=True)
            return fallback_path


@dataclass
class UISettings:
    """User interface configuration."""
    window_width: int = 500
    window_height: int = 570
    window_title: str = "mc-vc"
    theme: str = "dark"
    color_theme: str = "green"
    icon_path: str = "icons/titlebar-icon.ico"


@dataclass
class PrefixConfig:
    """Configuration for a single prefix."""
    hotkey: str
    prefix: str
    label: str


@dataclass
class ApplicationSettings:
    """Main application configuration."""
    audio: AudioSettings
    ui: UISettings
    default_auto_send: bool = True
    default_prefix_configs: Dict[str, PrefixConfig] = None
    
    def __post_init__(self):
        if self.default_prefix_configs is None:
            self.default_prefix_configs = {
                'prefix1': PrefixConfig(
                    hotkey='G', prefix='!', label='Prefix 1'
                ),
                'prefix2': PrefixConfig(
                    hotkey='L', prefix='', label='Prefix 2'
                ),
                'prefix3': PrefixConfig(
                    hotkey='P', prefix='/pc', label='Prefix 3'
                )
            }


app_settings = ApplicationSettings(
    audio=AudioSettings(),
    ui=UISettings()
)

