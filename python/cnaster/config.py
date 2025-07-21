from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


# TODO
_global_config = None


def set_global_config(config):
    global _global_config
    _global_config = config


def get_global_config():
    return _global_config


class JSONConfig:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = JSONConfig(v)

            setattr(self, k, v)

    def __iter__(self):
        return iter(
            xx for xx in dir(self) if (xx != "from_file") and not xx.startswith("_")
        )

    def __str__(self):
        return json.dumps(self, indent=4)

    @classmethod
    def from_file(cls, path):
        with Path.open(path, "r") as f:
            # Remove comments if present (JSON standard does not allow them)
            lines = [line for line in f if not line.strip().startswith("//")]
            d = json.loads("".join(lines))
        return cls(d)


class YAMLConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, YAMLConfig(value))
            elif isinstance(value, str) and value.lower() == "none":
                setattr(self, key, None)
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"YAMLConfig(\n{self._format_dict(self.__dict__, indent=2)}\n)"

    def _format_dict(self, d: Dict[str, Any], indent: int = 0) -> str:
        items = []
        space = " " * indent

        for key, value in d.items():
            if isinstance(value, YAMLConfig):
                formatted_value = (
                    f"\n{self._format_dict(value.__dict__, indent + 2)}\n{space}"
                )
            elif isinstance(value, dict):
                formatted_value = (
                    f"{{\n{self._format_dict(value, indent + 2)}\n{space}}}"
                )
            elif isinstance(value, str):
                formatted_value = f"'{value}'"
            else:
                formatted_value = repr(value)
            items.append(f"{space}{key}: {formatted_value}")

        return ",\n".join(items)

    @classmethod
    def from_file(cls, config_path: str | Path) -> YAMLConfig:
        with Path.open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        config = cls(config_dict)

        logger.info(f"Read configuration:\n{config}")

        return config
