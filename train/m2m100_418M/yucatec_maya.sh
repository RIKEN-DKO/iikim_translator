#!/bin/bash
#This one train a single model for all pairs from spanish to native langs
# Directory containing the YAML configuration files
config_file="configs/m2m100_418M/yucatec_maya.yaml"

python -m spanat.train_m2m --config_file $config_file
