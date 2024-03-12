import random
import string

import yaml


class YamlConfigParser:
    def __init__(self, config_path=None):
        self.config = None

        if config_path is None:
            run_name = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
            self.output_dir = f"run-{run_name}"
            self.report_to = "none"
        else:
            with open(config_path) as yaml_file:
                self.config = yaml.safe_load(yaml_file)
            self.output_dir = self.config["output_dir"]
            self.report_to = self.config["report_to"]
