#!/bin/bash

# Define the source language
source_lang="spanish"
model="assets/t5-small/otomay"
# Define the list of target languages
languages=("chol" "maya" "mazatec" "mixtec" "otomi")
# languages=("nahuatl")
# languages=("aymara" "quechua" "chol" "maya")

# Loop through the languages array
for target_lang in "${languages[@]}"; do
    echo "Testing source language: $source_lang with target language: $target_lang"
    python -m spanat.testT5 --model_path $model --base_data_dir data/indi_langs_clean --source_lang $source_lang --target_lang $target_lang --append_mode a
done
