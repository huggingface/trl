from typing import Any, Callable, List, Optional, Union

import torch
from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..core import set_seed
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper


class BestOfNSampler(object):
    def __init__(
        self,
        model: PreTrainedModelWrapper,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        best_of_lot: Callable[
            [List[str]], str
        ],  # callable that takes a list of generated texts and returns the best one
        sample_size: int = 4,
        length_sampler: Any = None,
        seed: Optional[int] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        if seed is not None:
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

        self.best_of_lot = best_of_lot
        self.length_sampler = length_sampler
        self.gen_config = generation_config
        self.sample_size = sample_size

    def generate(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        skip_special_tokens: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **generation_kwargs,  # way to override generation config
    ):
        if isinstance(query_tensor, torch.Tensor) and query_tensor.ndim == 1:
            query_tensor = torch.tensor(query_tensor).unsqueeze(0)
        elif isinstance(query_tensor, List):
            # assuming homogenous
            if not isinstance(query_tensor[0], torch.Tensor):
                query_tensor = [torch.tensor(query).reshape((1, -1)) for query in query_tensor]

        result = []

        for query in query_tensor:
            queries = query.repeat((self.sample_size, 1))
            output = self.model.generate(
                queries.to(device),
                max_new_tokens=self.length_sampler(),
                generation_config=self.gen_config,
                **generation_kwargs,
            ).squeeze()
            output = self.tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)
            result.append(self.best_of_lot(output))

        return result
