import math
from dataclasses import dataclass
from typing import Tuple, Literal
from transformers import AutoModel, AutoTokenizer

@dataclass
class ChatMlSpecialTokens:
  """Dataclass for special tokens used in ChatML, including system, user, assistant, bos, eos, and pad tokens."""
  bos_token: str = "<|im_start|>"
  eos_token: str = "<|im_end|>"
  pad_token: str = "<|im_end|>"
  
  @property
  def system(self):
    return f'{self.bos_token}system'
  
  @property
  def user(self):
    return f'{self.bos_token}user'

  @property
  def assistant(self):
    return f'{self.bos_token}assistant'

  @property
  def chat_template(self):
    return (
    "{% for message in messages %}"
    f"{{'{self.bos_token}' + message['role'] + '\n' + message['content'] + eos_token + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    f"{{ '{self.assistant}\n' }}"
    "{% endif %}"
)
    
FORMAT_MAPPING = {
  "chatml": ChatMlSpecialTokens
  }


def setup_chat_format(model: AutoModel, tokenizer: AutoTokenizer, format: Literal["chatml"]="chatml", resize_to_multiple_of_32=True) -> Tuple[AutoModel, AutoTokenizer]: 
    """
    Setup chat format by adding special tokens to the tokenizer, setting the correct format, and extending the embedding layer of the model based on the new special tokens.

    Args:
      model (AutoModel): The model to be modified.
      tokenizer (AutoTokenizer): The tokenizer to be modified.
      format (Literal["chatml"], optional): The format to be set. Defaults to "chatml".
      resize_to_multiple_of_32 (bool, optional): Whether to resize the model's input to a multiple of 32. Defaults to True.
    Returns:
      model (AutoModel): The modified model.
      tokenizer (AutoTokenizer): The modified tokenizer.
    """
    # get correct format
    chat_format = FORMAT_MAPPING[format]()

    # set special tokens and them 
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token
    tokenizer.bos_token = chat_format.bos_token
    tokenizer.add_special_tokens(
            {"additional_special_tokens": [chat_format.bos_token, chat_format.eos_token]}
        )
    # set chat format for tokenizer
    tokenizer.chat_template = chat_format.chat_template

    # resize embedding layer
    new_embedding_len = math.ceil(len(tokenizer) / 32) * 32 if resize_to_multiple_of_32 else len(tokenizer) 
    model.resize_token_embeddings(new_embedding_len)

    return model, tokenizer