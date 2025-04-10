import os
import logging

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "dummy-key"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting Aider")

coder = Coder.create(
    main_model = Model("openai/Qwen2.5-Coder-32B-Instruct"),
    io = InputOutput(yes=True)
)
coder.run("Remove all the logging crap in test.py")
