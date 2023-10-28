import typing
from itertools import chain

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_conflictEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    prompts
) -> typing.Dict:

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record[x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["prompt"].format(subject)]
    generation_prompts = [prompt.format(subject) for prompt in prompts[record["relation_id"]]]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        generation_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(generation_prompts))]
    ]
    # Flatten all the evaluated prefixes into one list.
    targets_res = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
    )
    # Unflatten the results again into a list of lists.
    ret = dict(
        rewrite = targets_res[0],
        generation = targets_res[1:]
    )

    return ret


def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
):

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {target_new}"
            for prefix in prefixes
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok = tok(f" {target_new}")["input_ids"]
    choice_a_len = len(a_tok)

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        
        cur_len = choice_a_len
        # Compute suffix probabilities
        all_probes = []
        for j in range(cur_len):
            cur_tok = a_tok[j]
            all_probes.append(-torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i] + j - 1, :], dim=0
            )[cur_tok].item())
        probs[i] = np.mean(all_probes)
        probs[i] = np.exp(-probs[i])
                    
    return probs.tolist()
