import ast
from pathlib import Path


def test_online_dpo_fix_param_name_delegates_to_shared_helper():
    source = Path("trl/experimental/online_dpo/online_dpo_trainer.py").read_text()
    tree = ast.parse(source)

    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "generation.vllm_sync_utils"
        and node.level == 3
        and any(alias.name == "fix_param_name_to_vllm" for alias in node.names)
        for node in tree.body
    )

    online_dpo_trainer = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OnlineDPOTrainer"
    )
    fix_method = next(
        node
        for node in online_dpo_trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "_fix_param_name_to_vllm"
    )
    return_stmt = next(node for node in fix_method.body if isinstance(node, ast.Return))

    assert isinstance(return_stmt.value, ast.Call)
    assert isinstance(return_stmt.value.func, ast.Name)
    assert return_stmt.value.func.id == "fix_param_name_to_vllm"
