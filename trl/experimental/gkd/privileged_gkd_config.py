# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field

from .gkd_config import GKDConfig


@dataclass
class PrivilegedGKDConfig(GKDConfig):
    """
    用于 PrivilegedSelfDistillTrainer 的配置。
    """

    privileged_key: str = field(
        default="privileged_messages",
        metadata={"help": "数据集中存放特权信息（PI）的字段名，值支持字符串或单组消息列表。"},
    )
    max_privileged_length: int = field(
        default=512,
        metadata={"help": "特权信息 tokenized 后的最大长度。"},
    )
    share_student_as_teacher: bool = field(
        default=True,
        metadata={"help": "是否让 teacher 与 student 共享同一套参数。"},
    )
    rollout_log_steps: int = field(
        default=0,
        metadata={"help": "rollout 日志打印间隔（按 step）。0 表示关闭。"},
    )
    rollout_log_samples: int = field(
        default=1,
        metadata={"help": "每次打印日志时展示的样本数。"},
    )
    rollout_log_max_new_tokens: int = field(
        default=64,
        metadata={"help": "teacher rollout 日志生成时的最大新 token 数。"},
    )
    rollout_log_max_chars: int = field(
        default=240,
        metadata={"help": "日志中每段文本的最大字符数，超过会截断。"},
    )
    debug_log_loss_steps: int = field(
        default=0,
        metadata={"help": "训练调试日志打印间隔（按 step）。0 表示关闭。"},
    )
    debug_log_grad_norm: bool = field(
        default=False,
        metadata={"help": "是否在调试日志中打印梯度 L2 范数与有梯度参数计数。"},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.max_privileged_length <= 0:
            raise ValueError("max_privileged_length 必须大于 0。")
        if self.rollout_log_steps < 0:
            raise ValueError("rollout_log_steps 不能为负数。")
        if self.rollout_log_samples <= 0:
            raise ValueError("rollout_log_samples 必须大于 0。")
        if self.rollout_log_max_new_tokens <= 0:
            raise ValueError("rollout_log_max_new_tokens 必须大于 0。")
        if self.rollout_log_max_chars <= 0:
            raise ValueError("rollout_log_max_chars 必须大于 0。")
        if self.debug_log_loss_steps < 0:
            raise ValueError("debug_log_loss_steps 不能为负数。")
