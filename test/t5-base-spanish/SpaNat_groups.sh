#!/bin/bash

# Define the source language
source_lang="spanish"



# model="assets/SpaNat_groups_t5-base-spanish/maya_chol"
# languages=("maya" "chol")

# model="assets/SpaNat_groups_t5-base-spanish/aymara_quechua"
# languages=("aymara" "quechua")
#--
# model="assets/SpaNat_groups_t5-base-spanish/raramuri_hñähñu_mixtec"
# languages=("raramuri" "hñähñu" "mixtec")

# model="assets/SpaNat_groups_t5-base-spanish/maya_chol_bribri"
# languages=("maya" "chol" "bribri")

# model="assets/SpaNat_groups_t5-base-spanish/otomi_mazatec"
# languages=("otomi" "mazatec")

# model="assets/SpaNat_groups_t5-base-spanish/shipibo_konibo_ashanika_nahuatl"
# languages=("shipibo_konibo" "ashanika" "nahuatl")


# Define an array of models and corresponding languages
models_and_languages=(
    # "assets/t5-base-spanish/SpaNat_groups/raramuri_hñähñu_mixtec raramuri hñähñu mixtec"
    # "assets/t5-base-spanish/SpaNat_groups/aymara_quechua aymara quechua"
    # "assets/t5-base-spanish/SpaNat_groups/maya_chol_bribri maya chol bribri"
    "assets/t5-base-spanish/SpaNat_groups/maya_chol maya chol"
    # "assets/t5-base-spanish/SpaNat_groups/otomi_mazatec otomi"
    # "assets/t5-base-spanish/SpaNat_groups/otomi_mazatec otomi mazatec"
    # "assets/t5-base-spanish/SpaNat_groups/shipibo_konibo_ashaninka_nahuatl shipibo_konibo ashaninka nahuatl"
)

# Iterate over each model and its languages
for item in "${models_and_languages[@]}"; do
    # Split the item into model and languages array
    read -r model languages <<< "$item"

    # Convert the languages string to an array
    IFS=' ' read -r -a languages_array <<< "$languages"

    # Iterate over each language for the current model
    for target_lang in "${languages_array[@]}"; do
        echo "Testing model: $model with target language: $target_lang"
        python -m spanat.testT5 --model_path "$model"\
        --base_data_dir data/indi_langs_clean\
        --source_lang $source_lang\
        --target_lang $target_lang\
        --append_mode a
    done
done
