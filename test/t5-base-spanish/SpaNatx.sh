#!/bin/bash

# Define the source language
source_lang="spanish"

# Define the list of target languages
languages=("aymara" "quechua" "chol" "maya" "raramuri" "shipibo_konibo" "hñähñu" "bribri" "chatino" "ashaninka" "wixarika" "guarani" "mazatec" "mixtec" "otomi" "nahuatl")
# languages=("nahuatl")
# languages=("aymara" "quechua" "chol" "maya")

# Loop through the languages array
for target_lang in "${languages[@]}"; do
    echo "Testing source language: $source_lang with target language: $target_lang"
    python -m translator.testT5 --model_path assets/t5-base-spanish/SpaNatx/$target_lang --base_data_dir data/indi_langs_clean --source_lang $source_lang --target_lang $target_lang
done
