# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from copy import deepcopy
from dataclasses import fields
from typing import Any, List

import yaml


class YamlConfigParser:
    def __init__(self, config_path: str = None, dataclasses: List[Any] = None):
        self.config = None

        if config_path is not None:
            with open(config_path) as yaml_file:
                self.config = yaml.safe_load(yaml_file)
        else:
            self.config = {}

        if dataclasses is None:
            dataclasses = []

        # We create a dummy training args to compare the values before / after
        # __post_init__
        # Here we import `TrainingArguments` from the local level to not
        # break TRL lazy imports.
        from transformers import TrainingArguments

        self._dummy_training_args = TrainingArguments(output_dir="dummy-training-args")

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
        from transformers import TrainingArguments

        dataclasses_copy = [deepcopy(dataclass) for dataclass in dataclasses]

        if len(self.config) > 0:
            for i, dataclass in enumerate(dataclasses):
                for data_class_field in fields(dataclass):
                    # Get the field here
                    field_name = data_class_field.name
                    field_value = getattr(dataclass, field_name)

                    if not isinstance(dataclass, TrainingArguments):
                        default_value = data_class_field.default
                    else:
                        default_value = (
                            getattr(self._dummy_training_args, field_name)
                            if field_name != "output_dir"
                            else field_name
                        )

                    default_value_changed = field_value != default_value

                    if field_value is not None or field_name in self.config:
                        if field_name in self.config:
                            # In case the field value is different from default, overwrite it
                            if default_value_changed and field_value != self.config[field_name]:
                                setattr(dataclasses_copy[i], field_name, self.config[field_name])
                            elif field_value is None and field_value != self.config[field_name]:
                                setattr(dataclasses_copy[i], field_name, self.config[field_name])
                            # Otherwise do nothing
                        elif default_value_changed:
                            setattr(dataclasses_copy[i], field_name, field_value)
        return dataclasses_copy

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
