#!/usr/bin/env bash
trap 'echo "Script interrupted"; exit' INT

datasets=("386" "387" "393")
lambda="0.001"
rotation="6d"
for dataset in "${datasets[@]}"; do
#  for lambda in "${lambdas[@]}"; do
#    for rotation in "${rotations[@]}"; do
      train_dataset="zjumocap_${dataset}_mono_half"
      checkpoint="./exp/${dataset}_2dgs_full"
      python train.py dataset="${train_dataset}" \
                      opt.lambda_nr_lipshitz_bound="${lambda}" \
                      dataset_name="${train_dataset}" \
                      +exp_dir=${checkpoint} \
                      rotation_representation="${rotation}" \
                      name="${dataset}_train_2dgs"
      test_dataset="zjumocap_${dataset}_mono_eval_novel"
      python render.py mode=test \
                       dataset.test_mode=view \
                       dataset="${test_dataset}" \
                       +exp_dir="${checkpoint}" \
                       +model.deformer.ablation=non-rigid \
                       rotation_representation="${rotation}" \
                       name="${dataset}_test_2dgs"
#    done
#  done
done