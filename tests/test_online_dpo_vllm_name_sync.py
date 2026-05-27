import ast
from pathlib import Path


VLLM_GENERATION_PATH = Path("trl/generation/vllm_generation.py")
ONLINE_DPO_PATH = Path("trl/experimental/online_dpo/online_dpo_trainer.py")
SHARED_HELPER_PATH = Path("trl/generation/vllm_sync_utils.py")

EXPECTED_RULE_SNIPPETS = [
    "self.model.config.architectures or []",
    '"Qwen3_5ForCausalLM"',
    '"Qwen3_5ForConditionalGeneration"',
    '"Qwen3VLForConditionalGeneration"',
    '"language_model.model."',
    '"language_model.lm_head."',
    '"visual."',
]
EXPECTED_INLINE_COMMENT = (
    "# Qwen3.5 text-only/VLM checkpoints use Hugging Face parameter prefixes that do not match the current\n"
    "        # vLLM runtime namespace, so we apply a narrow TRL-side compatibility shim during weight sync."
)


def _get_fix_method_source(path: Path) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    trainer_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    fix_method = next(
        node for node in trainer_class.body if isinstance(node, ast.FunctionDef) and node.name == "_fix_param_name_to_vllm"
    )
    return ast.get_source_segment(source, fix_method)


def test_weight_sync_logic_stays_inline_in_call_sites():
    assert not SHARED_HELPER_PATH.exists()

    for path in [VLLM_GENERATION_PATH, ONLINE_DPO_PATH]:
        source = path.read_text()
        assert "vllm_sync_utils" not in source


def test_vllm_generation_fix_param_name_inlines_qwen35_remap_rules():
    source = _get_fix_method_source(VLLM_GENERATION_PATH)

    for snippet in EXPECTED_RULE_SNIPPETS:
        assert snippet in source


def test_online_dpo_fix_param_name_inlines_qwen35_remap_rules():
    source = _get_fix_method_source(ONLINE_DPO_PATH)

    for snippet in EXPECTED_RULE_SNIPPETS:
        assert snippet in source


def test_duplicated_qwen35_inline_comments_match_word_for_word():
    generation_source = _get_fix_method_source(VLLM_GENERATION_PATH)
    online_dpo_source = _get_fix_method_source(ONLINE_DPO_PATH)

    assert EXPECTED_INLINE_COMMENT in generation_source
    assert EXPECTED_INLINE_COMMENT in online_dpo_source
