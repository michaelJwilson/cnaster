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
    assert config is None or isinstance(
        config, YAMLConfig
    ), "Global config must be None or a YAMLConfig instance"

    global _global_config
    _global_config = config


def get_global_config():
    if _global_config is None:
        logger.warning("cnaster config has not been defined.")

    else:
        assert isinstance(
            _global_config, YAMLConfig
        ), "Global config must be None or a YAMLConfig instance"

    return _global_config


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

        return config

    def over_ride(self, over_rides):
        if over_rides:
            for over_ride in over_rides:
                if "=" not in over_ride:
                    logger.warning(f"Provided over ride could not be resolved: {over_ride}")
                    raise ValueError

                key_path, value = over_ride.split("=", 1)
                keys = key_path.split(".")

                # NB find the correct sub-instance.
                obj = self
                
                for k in keys[:-1]:
                    obj = getattr(obj, k, None) if hasattr(obj, k) else obj.get(k)
                    
                final_key = keys[-1]
                
                if hasattr(obj, final_key):
                    setattr(obj, final_key, value)
                    logger.info(f"Config over ride: {key_path} = {value}")
                elif isinstance(obj, dict):
                    obj[final_key] = value
                    logger.info(f"Config over ride: {key_path} = {value}")
                else:
                    logger.warning(f"Cannot set config.{key_path}, skipping override.")

    def issue_warnings(self):
        if self.annotation.clone_label is not None:
            logger.warning(f"Assuming known clone labels")
        if not self.phasing.run:
            logger.warning(f"Assuming no baf-based phasing")
        if int(self.hmrf.n_clones_rdr) == 1:
            logger.warning(f"Assuming no rdr-based clone identification")
        if not self.hmrf.np_merge:
            logger.warning(f"Assuming no Neyman-Pearson model testing")
        if self.hmrf.fixed_assignment:
            logger.warning(f"Assuming fixed assignment")
            

            

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
