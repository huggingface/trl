# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of https://github.com/pytorch/torchtune.

import contextlib
from typing import Union
from warnings import warn

import psutil
import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks


class OffloadActivations(saved_tensors_hooks):
    """Context manager under which activation tensors created in the forward pass will be offloaded.

    Enable the memory efficiency technique of activation offloading, where activations bigger than
    min_offload_size bytes will be offloaded to CPU in the forward and brought back in the backward.
    This is in contrast to maintaining the activation on GPU VRAM throughout the program.

    This manager contains the option of using one additional CUDA stream to handle the communication
    between CUDA and CPU, which is intended to overlap with the default computation stream to improve
    runtime. We designed synchronization with a few heuristics for optimizing the tradeoff between
    runtime vs memory usage.

    Args:
        use_pin_memory (bool): Whether or not the offloaded Tensor will be placed in pinned
            memory on the CPU. Pinned memory allows the Tensor to be moved back onto GPU more quickly
            but is a limited resource. Default: True.

        use_streams (bool): Whether or not to use streams for performance optimization where
            the communications get overlapped with the computation. Requires a torch build
            after torch-2.5.0.]. Default: True.

        max_fwd_stash_size (int): The maximum size of the forward stash, or the maximum number of
            consecutive activations to keep alive during the forward pass. This number must be at
            least 1. Keeping alive more activations will potentially allow more overlap between the
            communication and compute streams at the cost of increasing memory usage. Keeping alive
            fewer activations will conserve memory, but may cause poor overlap between the streams,
            increasing runtime. Default: 5.

        min_offload_size (int): The minimum number of bytes a Tensor must be in order to qualify
            for offloading. If the tensor is too small, we do not want to waste bandwidth and resources
            moving it to CPU and back. Default: 1024 bytes.

    Raises:
        ValueError: if max_fwd_stash_size is not at least 1.

    Example:
        >>> with OffloadActivations():
        >>>     logits = model(inputs)
        >>> loss = ...
        >>> loss.backward()
    """

    def __init__(
        self,
        use_pin_memory: bool = True,
        use_streams: bool = True,
        max_fwd_stash_size: int = 5,
        min_offload_size: int = 1024,
    ) -> None:

        self.use_streams: bool = use_streams

        self.min_tensor_size_bytes = (
            min_offload_size  # we don't want to bother with small tensors
        )
        self.tracker = (
            {}
        )  # tensor_id => (new_tensor, if_modified)  ---> track what saved/offloaded tensors are where
        self.tensor_id: int = 0
        self.is_first_forward_call = True
        self.is_first_backward_call = True
        self.is_first_forward_pass = True

        # managing cpu memory
        self.use_pin_memory: bool = use_pin_memory
        self.virtual_memory_safe_pct = (
            60  # we should not exceed this percentage of memory
        )

        self.s0 = torch.cuda.default_stream()  # comp stream

        # for streaming
        if self.use_streams:
            self.s1 = torch.cuda.Stream()  # comms stream
            self.fwd_stash = {}  # tensor_id => (activation, ev1)
            if max_fwd_stash_size < 1:
                raise ValueError(
                    f"max_fwd_stash_size should be at least 1 but is {max_fwd_stash_size}"
                )
            self.max_fwd_stash_size = max_fwd_stash_size
            self.bwd_tensor_stash = {}  # tensor_id => activation
            self.bwd_ev_stash = {}  # tensor_id => ev0
            self.curr_graph_id = None
            self.curr_autograd_node = None

        # -------- platform util functions -------- #
        def verify_sufficient_virtual_memory():
            curr_pct = get_cpu_ram_pct()
            if curr_pct > self.virtual_memory_safe_pct:
                warn(
                    f"***** WARNING: {curr_pct=}% > {self.virtual_memory_safe_pct=}% of virtual memory used"
                )

        def get_cpu_ram_pct() -> float:
            # get the percentage of memory used by the system
            return psutil.virtual_memory().percent

        def get_tensor_id() -> int:
            # create a unique id for each tensor we are managing
            self.tensor_id += 1
            return self.tensor_id

        def get_num_bytes_tensor(x: torch.Tensor) -> int:
            # get the number of bytes in a tensor, for memory management purposes
            return (
                x.element_size() * x.nelement()
            )  # x.element_size() * x._base_storage().nbytes()

        # -------- core pack / unpack work -------- #
        def pack_tensor(activation: torch.Tensor) -> int:
            # activations are passed in during forward pass - from here we take over and return a unique id
            if self.is_first_forward_call:
                assert (
                    len(self.tracker) == 0
                ), "backward pass should have cleared tracker of all tensors"

                # set training phase trackers
                self.is_first_forward_call = False
                self.is_first_backward_call = True

            # query for basic tensor info
            num_bytes = get_num_bytes_tensor(activation)
            tensor_id = get_tensor_id()

            # only offload hefty bois if they're activations (our heuristic for that is to
            # check if they're not params or buffers)!
            if num_bytes >= self.min_tensor_size_bytes and (
                not isinstance(activation, torch.nn.Parameter)
                and not isinstance(activation, torch.nn.Buffer)
            ):
                if self.use_streams:
                    # First, sync back and dereference previously offloaded tensors
                    # as the offloading should be done sufficiently long ago.
                    for id in [k for k in self.fwd_stash.keys()]:
                        if id <= tensor_id - self.max_fwd_stash_size:
                            _, ev = self.fwd_stash[id]
                            self.s0.wait_event(ev)
                            del self.fwd_stash[id]
                        else:
                            break

                    # Sync in, offload, and add an event to sync back later
                    self.s1.wait_stream(self.s0)

                stream = self.s1 if self.use_streams else self.s0
                with torch.cuda.stream(stream):
                    try:
                        cpu_tensor = torch.empty_like(
                            activation, pin_memory=self.use_pin_memory, device="cpu"
                        )
                    except NotImplementedError as e:
                        if (
                            isinstance(activation, NF4Tensor)
                            and torchao.__version__ < "0.6.0.dev20240917"
                        ):
                            raise RuntimeError(
                                "Offloading NF4Tensors requires torchao-0.6.0.dev20240917 or later"
                            ) from e
                        raise e
                    cpu_tensor.copy_(activation, non_blocking=True)
                    self.tracker[tensor_id] = (
                        cpu_tensor,
                        True,
                    )  # True = (in future) modified

                if self.use_streams:
                    event = self.s1.record_event()

                    # Stash to keep activation alive til s1 is done
                    self.fwd_stash[tensor_id] = (activation, event)
            else:
                self.tracker[tensor_id] = (
                    activation,
                    False,
                )  # False = not modified, tensor is as is

            return tensor_id

        def unpack_tensor_single_stream(unpack_tensor_id: int) -> torch.Tensor:
            # backward pass - we are called with the tensor_id, which
            # we will use to retrieve the saved/offloaded tensor
            if self.is_first_backward_call:
                if self.is_first_forward_pass:
                    self.is_first_forward_pass = False
                    if self.use_pin_memory:
                        verify_sufficient_virtual_memory()

                self.is_first_backward_call = False
                self.is_first_forward_call = True

            assert (
                unpack_tensor_id in self.tracker
            ), f"untracked tensor with id {unpack_tensor_id}"

            maybe_gpu_tensor, modified = self.tracker[unpack_tensor_id]
            if modified:
                gpu_tensor = maybe_gpu_tensor.to("cuda", non_blocking=True)
                maybe_gpu_tensor = gpu_tensor

            # clear tensor from tracking
            del self.tracker[unpack_tensor_id]
            return maybe_gpu_tensor

        def unpack_tensor_with_streams(unpack_tensor_id: int) -> torch.Tensor:
            # backward pass - we are called with the tensor_id, which
            # we will use to retrieve the saved/offloaded tensor
            if self.is_first_backward_call:
                self.curr_graph_id = torch._C._current_graph_task_id()

                def wait_and_del_remaining_references() -> None:
                    for id in [k for k in self.bwd_tensor_stash.keys()]:
                        event = self.bwd_ev_stash[id]
                        self.s1.wait_event(event)
                        del self.bwd_tensor_stash[id]

                # Register a callback to the end of autograd to clean everything up
                torch.autograd.variable.Variable._execution_engine.queue_callback(
                    wait_and_del_remaining_references
                )

                if self.is_first_forward_pass:
                    self.is_first_forward_pass = False
                    if self.use_pin_memory:
                        verify_sufficient_virtual_memory()

                self.is_first_backward_call = False
                self.is_first_forward_call = True

            assert (
                unpack_tensor_id in self.tracker
            ), f"untracked tensor with id {unpack_tensor_id}"

            maybe_gpu_tensor, modified = self.tracker[unpack_tensor_id]
            if modified:
                # Get data on the current autograd node
                graph_id = torch._C._current_graph_task_id()
                node = torch._C._current_autograd_node()
                prev_node_ids = []

                # If we're on a new node, mark prev node's tensors to be freed later
                if graph_id == self.curr_graph_id and self.curr_autograd_node != node:
                    self.curr_autograd_node = node
                    prev_node_ids = [id for id in self.bwd_tensor_stash.keys()]

                brought_back_from_cpu = True
                if unpack_tensor_id in self.fwd_stash:
                    maybe_gpu_tensor = self.fwd_stash[unpack_tensor_id][0]
                    brought_back_from_cpu = False
                else:
                    # Kick off the process to bring tensors back
                    with torch.cuda.stream(self.s1):
                        gpu_tensor = maybe_gpu_tensor.to("cuda", non_blocking=True)
                        maybe_gpu_tensor = gpu_tensor

                    # Tell comp stream to wait for the info to be loaded before executing
                    self.s0.wait_stream(self.s1)

                    # Stash the tensor to keep memory alive until compute stream is complete
                    self.bwd_tensor_stash[unpack_tensor_id] = maybe_gpu_tensor

                    # Note: [Track views of the unpacked]
                    # Why do we get the use count of the unpacked tensor here? We want an
                    # initial count to compare to later, during the post-hook of the
                    # backward node, when we need to decide whether we're allowed to free
                    # the tensor yet. In what obscure cases must we delay freeing the
                    # tensor (and thus call record_stream)?
                    # 1. Any of the outputs of the backward node is a view of the unpacked
                    #    tensor.
                    # 2. In the case that this unpacked tensor will be used in a
                    #    checkpointed region, if one of the recomputed saved tensors ends
                    #    up as a view of the unpacked tensor.
                    # 3. The user abuses the system somehow and manually relies on the
                    #    unpacked tensor to exist after the backward node has executed.
                    storage_refcount = torch._C._storage_Use_Count(
                        maybe_gpu_tensor.untyped_storage()._cdata
                    )

                def hook(outputs, inputs):
                    # create events for the current node inputs/outputs if they were streamed in
                    if brought_back_from_cpu:
                        # See Note: [Track views of the unpacked]
                        # IF any of the outputs is a view of the tensor, OR if a view of
                        # the tensor has been saved as a part of checkpoint's recompute
                        # process, OR the user has abusedly incurred a reference on the
                        # unpacked tensor, THEN the tensor might be used later and we
                        # cannot presume to delete it after only the current node is
                        # done! So we use our frenemy, record_stream, to ensure the
                        # Tensor stays unmessed with until it's done getting used in the
                        # compute stream (s0 here). Note that the con here is we introduce
                        # non-deterministic (thus higher) memory usage, but this case
                        # should not happen often.
                        unpacked_tensor = self.bwd_tensor_stash[unpack_tensor_id]
                        if (
                            torch._C._storage_Use_Count(
                                unpacked_tensor.untyped_storage()._cdata
                            )
                            > storage_refcount
                        ):
                            unpacked_tensor.record_stream(self.s0)
                            del self.bwd_tensor_stash[unpack_tensor_id]
                        else:
                            event = self.s0.record_event()
                            self.bwd_ev_stash[unpack_tensor_id] = event

                    # if there are still things in the fwd_stash, get rid of them as we're in bwd now
                    for id in [k for k in self.fwd_stash.keys()]:
                        _, ev = self.fwd_stash[id]
                        self.s0.wait_event(ev)
                        del self.fwd_stash[id]

                    # wait on prev node's events and del those
                    for id in prev_node_ids:
                        event = self.bwd_ev_stash[id]
                        self.s1.wait_event(event)
                        del self.bwd_tensor_stash[id]

                    return outputs

                node.register_hook(hook)

            # clear tensor from tracking
            del self.tracker[unpack_tensor_id]
            return maybe_gpu_tensor

        unpack_tensor = (
            unpack_tensor_with_streams
            if self.use_streams
            else unpack_tensor_single_stream
        )
        super().__init__(pack_tensor, unpack_tensor)


class NoOpManager(saved_tensors_hooks):
    """
    A saved_tensors_hook manager used to disable any other saved_tensors_hook manager
    applied before. This relies on the behavior that only the most recently registered
    saved_tensors_hook will run.

    One example usage is to opt a local region of code out of activations offloading,
    which is usually applied globally to best track state.
    """

    def __init__(self) -> None:
        def noop(tensor):
            return tensor

        super().__init__(noop, noop)


def get_act_offloading_ctx_manager(
    model: nn.Module, enable_activation_offloading: bool
) -> Union[OffloadActivations, contextlib.nullcontext]:
    """Returns the activation offloading context manager for the model, which will be
    a null context if enable_activation_offloading is False.

    If activation offloading is enabled, we return the OffloadActivations context manager.
    If activation offloading is disabled, we return a NoOpManager context manager.

    Args:
        model (nn.Module): the model to wrap with the activation offloading context manager.
        enable_activation_offloading (bool): whether or not to enable activation offloading
            for the model.

    Returns:
        contextlib.ContextDecorator: the activation offloading context manager for the model.
    """
    if enable_activation_offloading:
        activations_handling_ctx = OffloadActivations()

        # Below is our hack to disable offloading the last output Linear in every
        # step, as the cost for offloading the activation and then soon after bringing
        # it back is expensive.
        output_head_detected = False
        noop_ctx = NoOpManager()

        # Try to get the actual model if it's wrapped
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "model"):
            model = model.model

        # Check for different types of output heads
        if hasattr(model, "output"):
            if isinstance(model.output, nn.Module):
                model.output.register_forward_pre_hook(
                    lambda *args: noop_ctx.__enter__()
                )
                model.output.register_forward_hook(
                    lambda *args: noop_ctx.__exit__(), always_call=True
                )
                output_head_detected = True
            elif isinstance(model.output, TiedLinear):
                model.output.linear.register_forward_pre_hook(
                    lambda *args: noop_ctx.__enter__()
                )
                model.output.linear.register_forward_hook(
                    lambda *args: noop_ctx.__exit__(), always_call=True
                )
                output_head_detected = True

        # Check for HuggingFace model output heads
        elif hasattr(model, "lm_head"):
            model.lm_head.register_forward_pre_hook(
                lambda *args: noop_ctx.__enter__()
            )
            model.lm_head.register_forward_hook(
                lambda *args: noop_ctx.__exit__(), always_call=True
            )
            output_head_detected = True

        elif hasattr(model, "decoder"):
            if hasattr(model.decoder, "output"):
                model.decoder.output.register_forward_pre_hook(
                    lambda *args: noop_ctx.__enter__()
                )
                model.decoder.output.register_forward_hook(
                    lambda *args: noop_ctx.__exit__(), always_call=True
                )
                output_head_detected = True

        if not output_head_detected:
            warn(
                "During activation offloading, no output head was detected. "
                "If your model has an output head, it will be offloaded. "
                "This usually greatly slows training, given the large vocabulary size. "
                "To change this behavior, set your output head as model.output and make it "
                "an nn.Module."
            )

    else:
        activations_handling_ctx = contextlib.nullcontext()

    return activations_handling_ctx
