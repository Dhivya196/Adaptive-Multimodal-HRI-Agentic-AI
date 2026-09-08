"""YAML configuration loader and parser for the HRI system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from src.common.exceptions import ConfigurationError
from src.common.logger import get_logger

logger = get_logger("ConfigLoader")

DEFAULT_CONFIG: Dict[str, Any] = {
    "vision": {
        "model_name": "yolo26n.pt",
        "device": "auto",
        "confidence_threshold": 0.45,
        "iou_threshold": 0.45,
        "target_classes": [
            "person", "bottle", "cup", "wine glass", "bowl",
            "banana", "apple", "sandwich", "orange", "chair",
            "couch", "dining table", "laptop", "cell phone", "book"
        ],
    },
    "camera": {
        "device_id": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
    },
    "spatial": {
        "sector_split": [0.33, 0.67],
        "near_area_ratio_threshold": 0.15,
        "medium_area_ratio_threshold": 0.05,
    },
    "visualizer": {
        "enable_display": True,
        "window_name": "HRI Multimodal Perception - Vision Agent",
        "show_spatial_sectors": True,
        "show_labels": True,
        "show_summary_banner": True,
    }
}


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges dict2 into dict1."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file, merging with default settings.
    If no path is provided, looks for configs/vision_config.yaml.
    """
    config = DEFAULT_CONFIG.copy()

    if config_path is None:
        default_yaml_path = Path(__file__).resolve().parent.parent.parent / "configs" / "vision_config.yaml"
        if default_yaml_path.exists():
            config_path = str(default_yaml_path)

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
                config = deep_merge(config, user_cfg)
                logger.info(f"Loaded configuration from: {config_path}")
        except Exception as e:
            logger.error(f"Error reading config file '{config_path}': {e}")
            raise ConfigurationError(f"Failed to parse config '{config_path}': {e}") from e
    else:
        if config_path:
            logger.warning(f"Config path '{config_path}' not found, using built-in defaults.")
        else:
            logger.info("Using default HRI configuration.")

    return config
