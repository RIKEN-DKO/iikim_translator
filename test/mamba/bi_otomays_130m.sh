#!/bin/bash

# Define the source language
source_lang="spanish"
model="assets/mamba/bi_otomays_130m"
# Define the list of target languages
languages=("chol" "maya" "mazatec" "mixtec" "otomi")
#last language was quechua
# Loop through the languages array
for target_lang in "${languages[@]}"; do
    echo "Testing source language: $source_lang with target language: $target_lang"
    cmd="python -m spanat.testmamba --model_path $model --base_data_dir data/otomay --source_lang $source_lang --target_lang $target_lang --append_mode a"
    # Print the command
    echo "Executing: $cmd"
    # Execute the command
    eval $cmd
done

./spanat/gather_metrics.sh $model