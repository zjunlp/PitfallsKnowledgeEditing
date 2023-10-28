import json
from numpy import mean
from pprint import pprint 
import os
import torch
import torch.nn.functional as F

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--res_dir", type=str, default="./GPT-J"
)
args = parser.parse_args()

RES_DIR = args.res_dir

datasets = ["coverage", "reverse", "composite"]
methods = ["FT", "MEND", "ROME", "MEMIT"]
debug = 5
result_dict = {}
for dataset in datasets:
    for method in methods:
        if os.path.exists(f"./{RES_DIR}/conflict_results/{method}_{dataset}.json"):
            with open(f"./{RES_DIR}/conflict_results/{method}_{dataset}.json") as fp:
                data = json.load(fp)
            CS = 0
            CM = 0
            if dataset == "composite":
                fixed = 0 
                fixed_o = 0  
            elif dataset == "coverage":
                S = 0
            for record in data:
                cs_pre = mean(record["CS_pre"]["generation"] + [record["CS_pre"]["rewrite"]])
                cs_post = mean(record["CS_post"]["generation"] + [record["CS_post"]["rewrite"]])
                cm_pre = mean(record["CM_pre"]["generation"] + [record["CM_pre"]["rewrite"]])
                cm_post = mean(record["CM_post"]["generation"] + [record["CM_post"]["rewrite"]])
                
                CS += 1 if cs_post > cs_pre else 0
                CM += max((cm_pre - cm_post) / cm_pre, -1)
                if dataset == "composite":
                    fact_0 = mean(record["fact_0"]["generation"] + [record["fact_0"]["rewrite"]])
                    fact_2 = mean(record["fact_2"]["generation"] + [record["fact_2"]["rewrite"]])
                    fact_o = mean(record["fact_o"]["generation"] + [record["fact_o"]["rewrite"]])
                    fixed += (fact_0 - fact_2) / fact_0
                    fixed_o += (fact_0 - fact_o) / fact_0
                elif dataset == "coverage":
                    s_post = mean(record["S_post"]["generation"] + [record["S_post"]["rewrite"]])
                    s_pre = mean(record["S_pre"]["generation"] + [record["S_pre"]["rewrite"]])
                    S += 1 if s_post > s_pre else 0
            CS /= len(data)
            CM /= len(data)
            if dataset == "composite":
                fixed /= len(data)
                fixed_o /= len(data)
                result_dict[f"{method}_{dataset}"] = dict(L=len(data),CS=CS,CM=CM,FFD=fixed)
            elif dataset == "coverage":
                S /= len(data)
                result_dict[f"{method}_{dataset}"] = dict(L=len(data),Succ=S,CS=CS,CM=CM)
            else:
                result_dict[f"{method}_{dataset}"] = dict(L=len(data),CS=CS,CM=CM)

for k, v in result_dict.items():
    print(f"{k}: {v}")

def js_div(p_output, q_output):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')
    p_output = F.normalize(torch.Tensor(p_output), p=1, dim=0)
    q_output = F.normalize(torch.Tensor(q_output), p=1, dim=0)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

datasets = ["easy", "hard"]
methods = ["FT", "MEND", "ROME", "MEMIT"]

result_dict = {}
for dataset in datasets:
    for method in methods:
        if os.path.exists(f"./{RES_DIR}/round_results/{method}_{dataset}.json"):
            with open(f"./{RES_DIR}/round_results/{method}_{dataset}.json") as fp:
                data = json.load(fp)
            with open(f"./{RES_DIR}/round_results/{dataset}_model.json") as fp:
                model_data = json.load(fp)
            D = 0
            IR = 0
            FR = 0
            success = 0
            for idx, record in enumerate(data):
                model_dt = model_data[idx]
                success_pre1 = mean(record["edit1"]["target_true"]["generation"] + [record["edit1"]["target_true"]["rewrite"]])
                success_pre2 = mean(record["edit2"]["target_true"]["generation"] + [record["edit2"]["target_true"]["rewrite"]])
                success_post1 = mean(record["edit1"]["target_new"]["generation"] + [record["edit1"]["target_new"]["rewrite"]])
                success_post2 = mean(record["edit2"]["target_new"]["generation"] + [record["edit2"]["target_new"]["rewrite"]])
                probs_gptj = mean(model_dt["model"]["target_new"]["generation"] + [model_dt["model"]["target_new"]["rewrite"]])
                fail_num = 0
                other_probs_pre = []
                other_probs_post = []
                for idx in range(len(record["edit2"]["others"])):
                    other_probs_pre.append(mean(model_dt["model"]["others"][idx]["generation"] + [model_dt["model"]["others"][idx]["rewrite"]]))
                    other_probs_post.append(mean(record["edit2"]["others"][idx]["generation"] + [record["edit2"]["others"][idx]["rewrite"]]))
                    if other_probs_pre[idx] > other_probs_post[idx]:
                        fail_num += 1
                IR += fail_num / len(record["edit2"]["others"])
                FR += 1 if fail_num / len(record["edit2"]["others"]) > 0.5 else 0
                success += 0.5 if success_post1 > success_pre1 else 0
                success += 0.5 if success_post2 > success_pre2 else 0
                D += js_div([success_post2]+other_probs_post, [probs_gptj]+other_probs_pre).item()
            success /= len(data)
            D /= len(data)
            IR /= len(data)
            FR /= len(data)
            result_dict[f"{method}_{dataset}"] = dict(L=len(data),Succ=success,D=D,IR=IR,FR=FR)

for k, v in result_dict.items():
    print(f"{k}: {v}")

datasets = ["easy", "hard"]
methods = ["MEMIT"]

result_dict = {}
for dataset in datasets:
    for method in methods:
        if os.path.exists(f"./{RES_DIR}/round_results/{method}_{dataset}_multi.json"):
            with open(f"./{RES_DIR}/round_results/{method}_{dataset}_multi.json") as fp:
                data = json.load(fp)
            with open(f"./{RES_DIR}/round_results/{dataset}_model.json") as fp:
                model_data = json.load(fp)
            D = 0
            IR = 0
            FR = 0
            success = 0
            for idx, record in enumerate(data):
                model_dt = model_data[idx]
                success_pre1 = mean(record["edit1"]["target_true"]["generation"] + [record["edit1"]["target_true"]["rewrite"]])
                success_pre2 = mean(record["edit2"]["target_true"]["generation"] + [record["edit2"]["target_true"]["rewrite"]])
                success_post1 = mean(record["edit1"]["target_new"]["generation"] + [record["edit1"]["target_new"]["rewrite"]])
                success_post2 = mean(record["edit2"]["target_new"]["generation"] + [record["edit2"]["target_new"]["rewrite"]])
                probs_gptj = mean(model_dt["gptj"]["target_new"]["generation"] + [model_dt["gptj"]["target_new"]["rewrite"]])
                fail_num = 0
                other_probs_pre = []
                other_probs_post = []
                for idx in range(len(record["edit2"]["others"])):
                    other_probs_pre.append(mean(model_dt["gptj"]["others"][idx]["generation"] + [model_dt["gptj"]["others"][idx]["rewrite"]]))
                    other_probs_post.append(mean(record["edit2"]["others"][idx]["generation"] + [record["edit2"]["others"][idx]["rewrite"]]))
                    if other_probs_pre[idx] > other_probs_post[idx]:
                        fail_num += 1
                IR += fail_num / len(record["edit2"]["others"])
                FR += 1 if fail_num / len(record["edit2"]["others"]) > 0.5 else 0
                success += 0.5 if success_post1 > success_pre1 else 0
                success += 0.5 if success_post2 > success_pre2 else 0
                D += js_div([success_post2]+other_probs_post, [probs_gptj]+other_probs_pre).item()
            success /= len(data)
            D /= len(data)
            IR /= len(data)
            FR /= len(data)
            result_dict[f"{method}_{dataset}_multi"] = dict(L=len(data),Succ=success,D=D,IR=IR,FR=FR)

for k, v in result_dict.items():
    print(f"{k}: {v}")