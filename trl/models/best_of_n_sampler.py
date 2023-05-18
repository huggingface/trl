import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable

from . import PreTrainedModelWrapper, AutoModelForCausalLMWithValueHead, SUPPORTED_ARCHITECTURES
from ..core import ( set_seed )
from transformers import Pipeline, pipeline, PreTrainedTokenizer, PreTrainedTokenizerFast, GenerationConfig, PipelineDataFormat

class BestOfNSampler(object):

    def __init__(
        self, 
        model: AutoModelForCausalLMWithValueHead, 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
        reward_pipeline: Pipeline, 
        sample_size:int = 4, 
        length_sampler: Any = None, 
        seed: Optional[int] = None, 
        generation_config: Optional[GenerationConfig] = None, 
        reward_kwargs: Dict[str, Any] = {}
    ) -> None:
    
        if seed != None:
            set_seed(seed)

        if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            raise ValueError(
                f"tokenizer must be a PreTrainedTokenizer or PreTrainedTokenizerFast, got {type(tokenizer)}"
            )
        if not isinstance(model, (SUPPORTED_ARCHITECTURES)):
            raise ValueError(
                f"model must be a PreTrainedModelWrapper, got {type(model)} - supported architectures are: {SUPPORTED_ARCHITECTURES}"
            )

        self.model = model
        self.tokenizer = tokenizer

        if reward_pipeline is None:
            raise ValueError("reward_pipeline must be provided")

        self.reward_pipeline = reward_pipeline
        self.length_sampler = length_sampler
        self.gen_config = generation_config
        self.reward_kwargs = reward_kwargs
        self.sample_size = sample_size

    def generate(self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        skip_special_tokens: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **generation_kwargs  # way to override generation config
    ):
        if not isinstance(query_tensor, List):
            query_tensor = [query_tensor]
        
        result = []

        for query in query_tensor:
            queries = query.repeat((self.sample_size, 1))
            output = self.model.generate(queries.to(device), max_new_tokens=self.length_sampler(), generation_config=self.gen_config, **generation_kwargs).squeeze()
            output = self.tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)
            scores = torch.tensor([output[0]["score"] for output in self.reward_pipeline(output, **self.reward_kwargs)])
            result.append(output[torch.argmax(scores)])

        return result