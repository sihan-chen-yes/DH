#!/usr/bin/env bash
trap 'echo "Script interrupted"; exit' INT

datasets=("386" "387" "393")

declare -A params
params=( ["pair1"]="0.0 0.0" ["pair2"]="1.0 0.001" )

for dataset in "${datasets[@]}"; do
  for pair in "${!params[@]}"; do
    IFS=' ' read -r lambda_xyz lambda_cov <<< "${params[$pair]}"

    train_dataset="zjumocap_${dataset}_mono_half"
    checkpoint="./exp/${dataset}_2dgs_full_aiap_ablation_${lambda_xyz}_${lambda_cov}"

    python train.py dataset="${train_dataset}" \
                    opt.lambda_aiap_xyz="${lambda_xyz}" \
                    opt.lambda_aiap_cov="${lambda_cov}" \
                    +exp_dir=${checkpoint} \

    test_dataset="zjumocap_${dataset}_mono_eval_novel"
    python render.py mode=test \
                     dataset.test_mode=view \
                     dataset="${test_dataset}" \
                     opt.lambda_aiap_xyz="${lambda_xyz}" \
                     opt.lambda_aiap_cov="${lambda_cov}" \
                     +exp_dir="${checkpoint}"
  done
done