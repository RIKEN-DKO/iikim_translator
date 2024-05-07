#!/bin/bash
#This one train a single model for all pairs from spanish to native langs
# Directory containing the YAML configuration files
config_file="configs/t5-base-spanish/mayan.yaml"
export TOKENIZERS_PARALLELISM=false
python -m spanat.trainT5 --config_file $config_file
