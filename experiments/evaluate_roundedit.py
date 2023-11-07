import os
import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit.baselines.ft import FTHyperParams, apply_ft_to_model
from memit.baselines.mend import MENDHyperParams, MendRewriteExecutor
from dataset import (
    RoundEditDataset
)
from experiments.py.eval_utils_conflictedit import evaluate_roundEdit
from memit.memit import MEMITHyperParams, apply_memit_to_model
from memit.rome import ROMEHyperParams, apply_rome_to_model
from memit.util import nethook
from memit.util.globals import *

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    dataset_size_limit: int,
    conserve_memory: bool,
    num_edits: int = 1,
    mode: str = ""
):
    data_dir = "./data/GPT2-XL" if "xl" in model_name.lower() else "./data/GPT-J"
    with open(f"{data_dir}/round_prompts.json") as fp:
        generation_prompts = json.load(fp)
    
    safe_model_name = model_name.replace("/", "_")
    if not os.path.exists(f"./results/{safe_model_name}/round_results"):
        os.makedirs(f"./results/{safe_model_name}/round_results")
    
    params_class, apply_algo = ALG_DICT[alg_name]
    out_file = f"./results/{safe_model_name}/round_results/{alg_name}_{mode}.json"
    
    hparams = params_class.from_json("memit" / HPARAMS_DIR / alg_name / hparams_fname)
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
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
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict()

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

        edited_model, weights_copy = apply_algo(
            model,
            tok,
            [
                edit_1(record)
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
            **etc_args,
        )

        edited_model_2, weights_copy_2 = apply_algo(
            edited_model,
            tok,
            [
                edit_2(record)
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
            **etc_args,
        )
        exec_time = time() - start
        print("Execution took", exec_time)

        # Evaluate new model
        start = time()
        for record in record_chunks:
            record["edit2"] = evaluate_roundEdit(
                edited_model_2,
                tok,
                edit_2(record),
                generation_prompts,
                record["true_objects"]
            )

            with torch.no_grad():
                for k, v in weights_copy_2.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")
            
            record["edit1"] = evaluate_roundEdit(
                model,
                tok,
                edit_1(record),
                generation_prompts
            )

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")
            
            all_results.append(record)
            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(all_results, f)

        # Restore original weights
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")

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
        "--alg_name",
        choices=["MEMIT", "ROME", "FT", "MEND"],
        default="ROME",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
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
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.dataset_size_limit,
        args.conserve_memory,
        num_edits=args.num_edits,
        mode=args.mode,
    )
