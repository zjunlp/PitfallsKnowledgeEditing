import os
import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import (
    RoundEditDataset
)
from experiments.py.eval_utils_conflictedit import evaluate_roundEdit

def main(
    model_name: Union[str, Tuple],
    dataset_size_limit: int,
    num_edits: int = 1,
    mode: str = ""
):
    data_dir = "./data/GPT2-XL" if "xl" in model_name.lower() else "./data/GPT-J"
    with open(f"{data_dir}/round_prompts.json") as fp:
        generation_prompts = json.load(fp)
    
    safe_model_name = model_name.replace("/", "_")
    if not os.path.exists(f"./{safe_model_name}/round_results"):
        os.makedirs(f"./{safe_model_name}/round_results")
    
    out_file = f"/{safe_model_name}/round_results/{mode}_model.json"
    
    if type(model_name) is str:
        print("Instantiating model")
        model_path = model_name
        model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
        
    # Load data
    ds = RoundEditDataset(f"{data_dir}/{mode}.json", tok=tok, size=dataset_size_limit)

    # Iterate through dataset
    all_results = []
    for record_chunks in chunks(ds, num_edits):
        start = time()
        edit_1 = lambda record: dict(
            prompt = record["edit"]["relation"]["prompt"],
            relation_id = record["edit"]["relation"]["id"],
            target_new = dict(
                str = record["edit"]["new_object"]["label"],
                id = record["edit"]["new_object"]["id"],
            ),
            target_true = dict(
                str = record["edit"]["object"]["label"],
                id = record["edit"]["object"]["id"],
            ),
            subject = record["edit"]["subject"]["label"]
        )
        edit_2 = lambda record: dict(
            prompt = record["edit"]["relation"]["prompt"],
            relation_id = record["edit"]["relation"]["id"],
            target_true = dict(
                str = record["edit"]["new_object"]["label"],
                id = record["edit"]["new_object"]["id"],
            ),
            target_new = dict(
                str = record["edit"]["object"]["label"],
                id = record["edit"]["object"]["id"],
            ),
            subject = record["edit"]["subject"]["label"]
        )
        for record in record_chunks:
            record["model"] = evaluate_roundEdit(
                model,
                tok,
                edit_2(record),
                generation_prompts,
                record["true_objects"]
            )
            
            all_results.append(record)
            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(all_results, f)

        # Restore original weights

        print("Evaluation took", time() - start)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--mode"
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.model_name,
        args.dataset_size_limit,
        num_edits=args.num_edits,
        mode=args.mode,
    )
