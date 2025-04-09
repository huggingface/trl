# We've modified trl's vllm-serve source code emulate OpenAI API compatability

import os
import tempfile

# Install in new environment, reqs don't work since trl is fixed to main on my fork
# This will be a requirement in the other repo, so it won't be an issue there
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput


os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "dummy-key"

with tempfile.TemporaryDirectory(dir=".") as temp_folder:
    original_dir = os.getcwd()
    os.chdir(temp_folder)
    
    coder = Coder.create(
        main_model=Model("openai/Qwen/Qwen2.5-Coder-1.5B-Instruct"),  # just a placeholder
        io=InputOutput(yes=True)
    )
    coder.run("Setup a simple snake game in python")
            
