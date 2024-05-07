#!/bin/bash

# Define the source language
source_lang="spanish"
model_path="assets/t5-base-spanish/SpaNat_order/last_model"
# Define the list of target languages
languages=("raramuri" "shipibo_konibo" "hñähñu" "bribri" "chatino" "ashaninka" "wixarika" "guarani" "mazatec" "mixtec" "otomi" "nahuatl" "aymara" "quechua" "chol" "maya")
# languages=("nahuatl")
# languages=("aymara" "quechua" "chol" "maya")

# Loop through the languages array
for target_lang in "${languages[@]}"; do
    echo "Testing source language: $source_lang with target language: $target_lang"
    python -m spanat.testT5 --model_path $model_path --base_data_dir data/indi_langs_clean --source_lang $source_lang --target_lang $target_lang --append_mode a
done
