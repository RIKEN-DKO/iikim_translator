#!/bin/bash

# Define the source language
source_lang="spanish"
model="assets/openLLama-SpaNat/"
# Define the list of target languages
languages=("raramuri" "shipibo_konibo" "hñähñu" "bribri" "chatino" "ashaninka" "aymara" "wixarika" "quechua" "guarani" "chol" "maya" "mazatec" "mixtec" "otomi" "nahuatl")
#last language was quechua
# Loop through the languages array
for target_lang in "${languages[@]}"; do
    echo "Testing source language: $source_lang with target language: $target_lang"
    python -m spanat.testLLama --model_path $model --base_data_dir data/indi_langs_clean --source_lang $source_lang --target_lang $target_lang
done
