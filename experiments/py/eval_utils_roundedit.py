import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_roundEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    prompts,
    true_objects = None
) -> typing.Dict:

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record[x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["prompt"].format(subject)]
    # paraphrase_prompts = record["paraphrase_prompts"]
    generation_prompts = [prompt.format(subject) for prompt in prompts[record["relation_id"]]]

    if true_objects != None:
        temp = []
        for obj in true_objects:
            temp.append(obj["label"])
        true_objects = temp
    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        generation_prompts,
        # paraphrase_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(generation_prompts))],
    ]
    # Flatten all the evaluated prefixes into one list.

    targets_res = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
        true_objects
    )
    rets = {}
    others = []
    for idx, line in enumerate(targets_res):
        if idx == 0:
            rets["target_true"] = dict(
                rewrite = line[0],
                generation = line[1:]
            )
        elif idx == 1:
            rets["target_new"] = dict(
                rewrite = line[0],
                generation = line[1:]
            )
        else:
            others.append(dict(
                rewrite = line[0],
                generation = line[1:]
            ))
    if len(others) > 0:
        rets["others"] = others

    return rets


def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
    true_objects = None
):
    targets = [target_true, target_new]
    if true_objects != None:
        true_objects.remove(target_new)
        targets += true_objects
    # target_true=target_true_all[0]
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in targets 
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    toks = [tok(f" {n}")["input_ids"] for n in targets]
    toks_len = [len(n) for n in toks]

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    softmax_res = np.zeros((logits.size(0) // len(prefixes), len(prefixes)), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_len = toks_len[i % len(targets)]
        score_list = np.zeros((cur_len,), dtype=np.float32)
        for j in range(cur_len):
            cur_tok = toks[i % len(targets)][j]
            scores = -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // len(targets)] + j - 1, :], dim=0
            )
            score_list[j] = scores[cur_tok].item()

        softmax_res[i % len(targets)][i // len(targets)] = np.exp(-score_list.mean())

    return softmax_res.tolist()
