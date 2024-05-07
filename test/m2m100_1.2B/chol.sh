#!/bin/bash

# Define the source language
source_lang="spanish"

model="assets/m2m100_418M/chol"

# Remove dev and test directories if they exist
rm -rf "$model/dev" "$model/test"

# Remove results.txt and table.csv files if they exist
rm -f "$model/results.txt" "$model/table.csv"


# Define the list of target languages
languages=("chol")
# languages=("nahuatl")
# languages=("aymara" "quechua" "chol" "maya")
# export CUDA_VISIBLE_DEVICES=0
# Loop through the languages array
for target_lang in "${languages[@]}"; do
    echo "Testing source language: $source_lang with target language: $target_lang"
    python -m spanat.test_m2m --model_path $model --base_data_dir data/indi_langs_clean --source_lang $source_lang --target_lang $target_lang --append_mode a
done

./spanat/gather_metrics.sh $model