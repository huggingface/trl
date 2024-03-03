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
from enum import Enum
from typing import Dict, Optional

from transformers import TrainingArguments


class FDivergenceType(Enum):
    REVERSE_KL = "reverse_kl"
    JS_DIVERGENCE = "js_divergence"
    ALPHA_DIVERGENCE = "alpha_divergence"


class FDivergenceConstants:
    ALPHA_DIVERGENCE_COEF_KEY = "alpha_divergence_coef"
    ALPHA_DIVERGENCE_COEF_DEFAULT = 1.0


@dataclass
class DPOConfig(TrainingArguments):
    """
    DPOConfig collects all training arguments related to the [`DPOTrainer`] class.
    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.
    Parameters:
        f_divergence_type (`FDivergenceType`, *optional*, defaults to `FDivergenceType.REVERSE_KL`):
            The type of f-divergence regularization function to compute divergence between policy and reference model. This argument is optional, defaults to `FDivergenceType.REVERSE_KL`.
        f_divergence_params (`Dict`, *optional*, defaults to `None`):
            The parameters of f-divergence regularization function, eg: the alpha parameter in alpha-divergence. This argument is optional, defaults to 'None'.
    """

    f_divergence_type: Optional[FDivergenceType] = FDivergenceType.REVERSE_KL
    """The type of f-divergence regularization function to compute divergence between policy and reference model, This argument is optional, defaults to `FDivergenceType.REVERSE_KL`."""
    f_alpha_divergence_coef: float = field(default=1.0, metadata={"help": "the alpha coef in alpha-divergence(u^-alpha) regularization function for DPO loss"})
    """The alpha coef in alpha-divergence(u^-alpha) regularization function for DPO loss."""
