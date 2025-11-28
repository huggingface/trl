from grader import math_equal
from parser import strip_string
import timeout_decorator
from collections import defaultdict, Counter
from utils import load_jsonl


@timeout_decorator.timeout(5)
def math_equal_timeout(pred, gt):
    try:
        return math_equal(pred, gt)
    except Exception as e:
        print("Timeout error:", e)
        return False


def group_pred(preds, strip=True, use_symbol=False):
    orginal_preds = preds
    if not use_symbol:
        if strip:
            preds = [strip_string(pred) for pred in preds]
        cnt = Counter(preds)
        majority = cnt.most_common(1)[0][0]
        groups = defaultdict(list)
        for idx, pred in enumerate(preds):
            groups[pred].append(idx)
        return groups, orginal_preds[groups[majority][0]]

    groups = defaultdict(list)
    for idx, pred in enumerate(preds):
        found_group = False
        if strip:
            pred = strip_string(pred)
        for group_pred in groups:
            try:
                if math_equal_timeout(pred, group_pred):
                    groups[group_pred].append(idx)
                    found_group = True
                    break
            except:
                continue
        if not found_group:
            groups[pred].append(idx)
    # get the key of the longest group
    majority = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)[0][0]
    majority = orginal_preds[groups[majority][0]]
    return groups, majority


def eval_rm_k_metrics(data_path, k=8):
    print(f"evaluating rm@{k}")
    data_list = load_jsonl(data_path)

    count, right_count = 0, 0
    for sample in data_list:
        assert len(sample['pred_score']) >= k, sample['data_source']
        pred_score = sample['pred_score'][:k]
        pred = sample['score'][:k]
        assert len(pred_score) == len(pred), f"{len(pred_score)}, {len(pred)}"

        rm_score = pred_score
        rm_score = [inner_score for score in rm_score for inner_score in score]
        assert len(rm_score) == len(pred), f"{len(rm_score)}, {len(pred)}"

        max_index = rm_score.index(max(rm_score))
        max_pred = pred[max_index]
        right_count += max_pred
        count += 1

    print(count)
    task_acc = right_count / count * 100
    print(f"acc: {task_acc:.1f}")
    return task_acc


def eval_maj_k_metrics(data_path, k=8):
    print(f"evaluating maj@{k}")

    data_list = load_jsonl(data_path)
    count, right_count = 0, 0
    for sample in data_list:
        assert len(sample['score']) >= k, sample
        groups, majority_pred = group_pred(sample['pred'][:k], strip=False, use_symbol=False)
        idx = groups[majority_pred][0]
        right_count += sample['score'][idx]
        count += 1

    task_acc = right_count / count * 100
    print(f"acc: {task_acc:.1f}")
    return task_acc


if __name__ == "__main__":
    data_path = "./data/eval_rm_maj_example/math_cot_100.jsonl"

    candidate = 8
    all_result = {}
    all_result[f'maj@{candidate}'] = eval_maj_k_metrics(data_path, k=candidate)
    all_result[f'rm@{candidate}'] = eval_rm_k_metrics(data_path, k=candidate)
    print(all_result)
