import copy
import json
import os
import shutil
from typing import Optional

import yaml

# ==================================================================================================

file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
train_config: Optional[dict] = None

# ==================================================================================================


def seconds_to_hours(secs: float) -> str:
    secs = int(secs)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    t = "{:d}:{:02d}:{:02d}".format(h, m, s)
    return t


# ==================================================================================================


def delete_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


# ==================================================================================================


def load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        content: dict = json.load(file)
    return content


# ==================================================================================================


def load_yaml_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        content: dict = yaml.load(file, Loader=yaml.SafeLoader)
    return content


# ==================================================================================================


def merge_dicts(source: dict, target: dict) -> dict:
    """Overwrite values in target dict recursivly with values from source dict"""

    for key, value in source.items():
        if isinstance(value, dict):
            if key in target:
                merged = merge_dicts(value, target[key])
            else:
                merged = value
            target[key] = merged
        else:
            target[key] = value

    return target


# ==================================================================================================


def nested_replace(target: dict, old_str: str, new_str: str) -> dict:
    """Replace a substring in the values of a nested dict"""

    result = copy.deepcopy(target)
    for key, value in target.items():
        if isinstance(value, dict):
            result[key] = nested_replace(value, old_str, new_str)
        elif isinstance(value, str):
            result[key] = value.replace(old_str, new_str)
        else:
            pass
    return result


# ==================================================================================================


def get_config() -> dict:
    """Get config or load it from the config files using template completion"""
    global train_config

    if train_config is not None:
        return train_config

    # Load user config if existing
    path = os.path.join(file_path + "../config/train_config.yaml")
    if os.path.exists(path):
        user_config = load_yaml_file(path)
    else:
        user_config = {}

    # Load template file
    path = os.path.join(file_path + "../config/train_config.template.yaml")
    tmpl_config = load_yaml_file(path)

    # Complete user config with template values if some values are missing
    # In this way we can add new config keys to the project without breaking older configs
    config = merge_dicts(user_config, tmpl_config)

    # Replace all language placeholders
    config = nested_replace(config, "{LANGUAGE}", config["language"])

    train_config = config
    return train_config


# ==================================================================================================


def load_alphabet(config: dict) -> list:
    """Load alphabet from file"""

    alphabet = load_json_file(config["alphabet_path"])
    return alphabet
