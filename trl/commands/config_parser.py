import os
from dataclasses import fields
from typing import Any, List

import yaml


class YamlConfigParser:
    # Some keys are processed by `TrainingArguments.__post_init__` and initialized there
    # this mapping ignores the processing of these keys if they are modified.
    _keys_ignore = {
        "accelerator_config": None,
        "report_to": "none",
        "fsdp_config": None,
        "logging_dir": None,
        "lr_scheduler_kwargs": None,
    }

    _not_supported_classes = [dict, list]

    def __init__(self, config_path: str = None, dataclasses: List[Any] = None):
        self.config = None

        if config_path is not None:
            with open(config_path) as yaml_file:
                self.config = yaml.safe_load(yaml_file)
        else:
            self.config = {}

        if dataclasses is None:
            dataclasses = []

        # Here we import `AcceleratorConfig` from the local level to not
        # break TRL lazy imports.
        try:
            from transformers.trainer_pt_utils import AcceleratorConfig

            self._not_supported_classes.append(AcceleratorConfig)
        except ImportError:
            ...

        self.parse_and_set_env()
        self.merge_dataclasses(dataclasses)

    def parse_and_set_env(self):
        if "env" in self.config:
            env_vars = self.config["env"]
            if isinstance(env_vars, dict):
                for key, value in env_vars.items():
                    os.environ[key] = str(value)
            else:
                raise ValueError("`env` field should be a dict in the YAML file.")

    def merge_dataclasses(self, dataclasses):
        for dataclass in dataclasses:
            for data_class_field in fields(dataclass):
                # Get the field here
                field_name = data_class_field.name
                field_value = getattr(dataclass, field_name)
                default_value = data_class_field.default

                default_value_changed = field_value != default_value

                if field_name in self._keys_ignore.keys():
                    if default_value_changed and not any(
                        isinstance(field_value, cls_to_test) for cls_to_test in self._not_supported_classes
                    ):
                        self.config[field_name] = field_value
                    continue

                if field_value is not None:
                    if data_class_field in self.config:
                        # In case the field value is different from default, overwrite it
                        if default_value_changed and field_value != self.config[field_name]:
                            self.config[field_name] = field_value
                        # Otherwise do nothing
                    elif default_value_changed:
                        self.config[field_name] = field_value

    def to_string(self):
        final_string = """"""
        for key, value in self.config.items():
            if isinstance(value, (dict, list)):
                if len(value) != 0:
                    value = str(value)
                    value = value.replace("'", '"')
                    value = f"'{value}'"
                else:
                    continue

            final_string += f"--{key} {value} "
        return final_string
