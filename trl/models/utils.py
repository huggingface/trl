import math
from dataclasses import dataclass
from typing import Tuple, Literal
from transformers import PreTrainedTokenizer, PreTrainedModel

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


def setup_chat_format(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, format: Literal["chatml"]="chatml", resize_to_multiple_of=2) -> Tuple[PreTrainedModel, PreTrainedTokenizer]: 
    """
    Setup chat format by adding special tokens to the tokenizer, setting the correct format, and extending the embedding layer of the model based on the new special tokens.

    Args:
      model (AutoModel): The model to be modified.
      tokenizer (AutoTokenizer): The tokenizer to be modified.
      format (Literal["chatml"], optional): The format to be set. Defaults to "chatml".
      resize_to_multiple_of (int, optional): Number to resize the embedding layer to. Defaults to 2.
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

    # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=resize_to_multiple_of)

    return model, tokenizer