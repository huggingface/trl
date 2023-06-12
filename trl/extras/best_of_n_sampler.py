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
        queries_to_scores: Callable[[List[str]], List[float]],
        length_sampler: Any,
        sample_size: int = 4,
        seed: Optional[int] = None,
        n_candidates: int = 1,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        r"""
        Initialize the sampler for best-of-n generation

        Args:
            model (`PreTrainedModelWrapper`):
                The pretrained model to use for generation
            tokenizer (`PreTrainedTokenizer` or `PreTrainedTokenizerFast`):
                Tokenizer associated with the pretrained model
            queries_to_scores (`Callable[[List[str]], List[float]]`):
                Callable that takes a list of generated texts and returns the associated reward scores
            length_sampler (`Any`):
                Sampler used to sample the length of the generated text
            sample_size (`int`):
                Number of samples to generate for each query
            seed (`int`, *optional*):
                Random seed used to control generation
            n_candidates (`int`):
                Number of candidates to return for each query
            generation_config (`GenerationConfig`, *optional*):
                Generation config passed to the underlying model's `generate` method.
                See `GenerationConfig` (https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig) for more details
        """
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

        self.queries_to_scores = queries_to_scores
        self.length_sampler = length_sampler
        self.gen_config = generation_config
        self.sample_size = sample_size
        self.n_candidates = n_candidates

    def generate(
        self,
        query_tensor: Union[List[int], torch.Tensor, List[torch.Tensor], List[List[int]]],
        skip_special_tokens: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **generation_kwargs,
    ) -> List[List[str]]:
        r"""
        Generate the best of n samples for input queries

        Args:
            query_tensor (`List[int]` or `torch.Tensor` or `List[torch.Tensor]` or `List[int]`):
                Query tensor or list of query tensors or list of plain int/list of int queries to generate from
            skip_special_tokens (`bool`):
                Whether to remove the special tokens from the output
            device (`str` or `torch.device`, *optional*):
                The device on which the model will be loaded
            **generation_kwargs (`dict`, *optional*):
                Additional keyword arguments passed along to the underlying model's `generate` method.
                This is used to override generation config

        Returns:
            List[List[str]]: A list of lists of generated texts
        """

        if isinstance(query_tensor, torch.Tensor) and query_tensor.ndim == 1:
            query_tensor = query_tensor.clone().detach().unsqueeze(0)
        elif isinstance(query_tensor, List):
            element_type = type(query_tensor[0])
            if element_type == int:
                query_tensor = torch.tensor(query_tensor).unsqueeze(0)
            elif element_type == torch.Tensor:
                query_tensor = [tensor.reshape((1, -1)) for tensor in query_tensor]
            else:
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
            scores = torch.tensor(self.queries_to_scores(output))
            output = [output[i] for i in scores.topk(self.n_candidates).indices]
            result.append(output)

        return result
