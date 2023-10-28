python3 -m experiments.evaluate_MLE \
    --alg_name=MEMIT \
    --model_name=gpt-j-6B \
    --hparams_fname=EleutherAI_gpt-j-6B.json \
    --dataset_size_limit 2500 \
    --mode=easy