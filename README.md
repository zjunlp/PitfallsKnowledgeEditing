# Unveiling the Pitfalls of Knowledge Editing for Large Language Models

![main](img/main.png)

## Table of Contents

- [Installation](#installation)
- [Dataset Format](#dataset-format)
- [Evaluation](#evaluation)
- [Summerization](#summerization)
- [How to Cite](#how-to-cite)

## Installation

Please use Python 3.9+ to get started, install `conda` and run:
```bash
conda create -n EditLLMs python=3.9.7
pip install -r requirements.txt
```

**Note:** We recommend `conda` for managing Python, CUDA, and PyTorch.

## Dataset Format

### ConflictEdit

Each dataset split in this part contains 2500 data, except 2000 data in `./data/GPT2-XL/composite_edit.json`. Take `reverse_edit.json` for example:

```json
{
    "rule": "Logical Rule", 
    "triples": [
        {
            "relation": {
                "id": "ID in WikiData",
                "prompt": "Prompt of Relation",
                "query": "Prompt of Relation in the query format",
                "label": "Relation Description"
            },
            "subject": {
                "id": "ID in WikiData",
                "label": "Entity Description"
            },
            "object": {
                "id": "ID in WikiData",
                "label": "Entity Description"
            }
        },                  // Triple 1
        "... Triple 2"
    ],
    "prerequisites": [],    // Tied Fact Dependency
    "type": "reverse",      // Edit Type
    "edits": [
        {
            "relation": "Same as above",
            "subject": "Same as above",
            "object": "Object to be edit",
            "new_object": "Target Object of editing"
        },                  // Edit 1
        "... Edit 2"
    ]
}
```

### RoundEdit

Each dataset split in this part contains 2500 data.Take `easy.json` for example:

```json
{
    "type": "1-N@RelationID",   // N means 1-n relation
    "edit": {
        "relation": "Same as above",
        "subject": "Same as above",
        "new_object": "Intermediate object in Round-Edit",
        "object": "Target object in Round-Edit"
    },
    "true_objects": [
        {
            "id": "ID in WikiData",
            "label": "Entity Description"
        },                      // True object 1
        "... True objects"
    ]
}
```

## Evaluation

### Knowledge Conflict

To evaluate Knowledge Conflict, simply utilize the scripts as:

```shell
bash run_conflictedit.sh
```

The dataset split can be changed by modified the `mode` in [`run_conflictedit.sh`](scripts/run_conflictedit.sh) and also the model type, hyperparameters and editing methods. The experimental results are written in `./{ModelName}/conflict_results/`

### Knowledge Distortion

To evaluate Knowledge Conflict, please follow the **Steps** as:

- **Step 1:** Obtain the results on the original model by running:

```shell
bash run_model.sh
```

- **Step 2:** Obtain the main results as:

```shell
bash run_roundedit.sh
```

- **Step 3:** Obtain the **Multi-Label Edit (MLE)** results as:

```shell
bash run_MLE.sh
```

The dataset split can be changed by modified the `mode` in each script and also the model type, hyperparameters and editing methods. The experimental results are written in `./{ModelName}/round_results/`


**Note:** We train MEND on our datasets and **the checkpoints will be released in the future**.

## Summerization

To summarize the results, you can use [`experiments/summarize.py`](experiments/summarize.py):

```bash
python3 -m experiments.summarize --res_dir=GPT-J
```

## How to Cite

```bibtex
@misc{li2023unveiling,
      title={Unveiling the Pitfalls of Knowledge Editing for Large Language Models}, 
      author={Zhoubo Li and Ningyu Zhang and Yunzhi Yao and Mengru Wang and Xi Chen and Huajun Chen},
      year={2023},
      eprint={2310.02129},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```