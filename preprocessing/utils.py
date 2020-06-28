import json
import os

# ==================================================================================================

repo_path: str
langdicts = None
file_path = os.path.dirname(os.path.realpath(__file__)) + "/"


# ==================================================================================================


def load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        content = json.load(file)
    return content


# ==================================================================================================


def load_config(config_dir: str, config_name: str) -> dict:
    """ Load config file with either '.json' or '.json.default' file endings """

    config_path = config_dir + config_name + ".json"
    if not os.path.exists(config_path):
        # User did not change config, load the default config
        config_path = config_dir + config_name + ".json.default"
    config = load_json_file(config_path)
    return config


# ==================================================================================================


def load_global_config() -> dict:
    path = file_path + "../data/config/"
    config = load_config(path, "global_config")
    return config


# ==================================================================================================


def get_langdicts() -> dict:
    """ Load the langdicts, or just return them if they are already loaded """
    global langdicts

    if langdicts is None:
        path = file_path + "../data/langdicts.json"
        langdicts = load_json_file(path)

    return langdicts
