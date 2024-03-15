import pandas as pd
import os
import yaml

ROOT = os.path.abspath(__file__)
for _ in range(2):
    ROOT = os.path.dirname(ROOT)

GLOBAL_CONFIG = os.path.join(ROOT, "configs", "global.yaml")


def load_global_config():
    with open(GLOBAL_CONFIG, "r") as rf:
        config_dict = yaml.load(rf, yaml.Loader)

    for holiday_key in config_dict["holiday_category"]:
        for yr_iter in config_dict[holiday_key]:
            config_dict[holiday_key][yr_iter] = pd.to_datetime(config_dict[holiday_key][yr_iter])

    return config_dict


def add_prefix_to_feature_names(pipeline_step_name: str, param_space_dict: dict):
    out_dict = {}
    for key, val in param_space_dict.items():
        out_dict[f"{pipeline_step_name}__{key}"] = val

    return out_dict
