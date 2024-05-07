#%%
# %load_ext autoreload
# %autoreload 2
#%%
import argparse
import yaml
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, 
                          Seq2SeqTrainingArguments, Seq2SeqTrainer, 
                          DataCollatorForSeq2Seq, EarlyStoppingCallback)
import os
from transformers import EarlyStoppingCallback
import math
# from .testT5 import test,generate_prompt 
from spanat.train_utils import (load_tokenized_data,
                                load_tokenized_data_per_lang,
                                BalancedMultiDataset)
#%%
# Setup command-line argument parsing
parser = argparse.ArgumentParser(description="Train a multilingual T5 model")
parser.add_argument('--config_file',
                    required=False, 
                    type=str, 
                    help='Path to the YAML configuration file',
                    default='/home/julio/repos/queTrans/configs/t5-base-spanish/SpaNat_uniform.yaml')
#debug
args = parser.parse_args()
# Load configuration from the provided YAML file
with open(args.config_file, 'r') as file:
    config = yaml.safe_load(file)

model_name = config['model_name']
base_dir = config['base_dir']
train_direction = config['train_direction']

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
def transform_dataset(dataset):
    # Remove the 'translation' field or transform it
    return dataset.remove_columns(["translation"])

def preprocess_function(examples, source_lang, target_lang):
    prefix = f"translate {source_lang} to {target_lang}: "
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]

    # Using the tokenizer's __call__ method for both encoding and padding
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    # Use tokenizer as target tokenizer for labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



def check_dataset_for_nan_inf(dataset, input_field, target_field):
    for i, example in enumerate(dataset):
        # Check input field
        input_values = example[input_field]
        if any(math.isnan(val) or math.isinf(val) for val in input_values):
            print(f"Found NaN or Inf in input at index {i}")

        # Check target field
        target_values = example[target_field]
        if any(math.isnan(val) or math.isinf(val) for val in target_values):
            print(f"Found NaN or Inf in target at index {i}")



#%%
# Combine all tokenized data for training and evaluation
target_langs = config['target_langs']

if config['uniform_sample']:
    tokenized_train_data = load_tokenized_data_per_lang(
        base_dir, target_langs, train_direction, 
        preprocess_function, transform_dataset)
    tokenized_eval_data = load_tokenized_data_per_lang(
        base_dir, target_langs, train_direction, 
        preprocess_function, transform_dataset,split='dev')
    
    # Assuming tokenized_train_data and tokenized_eval_data are dictionaries with language pair datasets
    combined_tokenized_train_data = BalancedMultiDataset(tokenized_train_data)
    combined_tokenized_eval_data = BalancedMultiDataset(tokenized_eval_data)
    
else:
    tokenized_train_data, tokenized_eval_data = load_tokenized_data(
        base_dir, target_langs, train_direction, preprocess_function, transform_dataset)
    combined_tokenized_train_data = concatenate_datasets(tokenized_train_data)
    combined_tokenized_eval_data = concatenate_datasets(tokenized_eval_data)
    # Shuffle the combined datasets
    combined_tokenized_train_data = combined_tokenized_train_data.shuffle(seed=42)
    combined_tokenized_eval_data = combined_tokenized_eval_data.shuffle(seed=42)
#%%


#%%
print("###Total size")
print('train:',combined_tokenized_train_data)
print('eval',combined_tokenized_eval_data)
# Example usage:
# Assuming 'input_ids' and 'labels' are your input and target fields in the dataset
# check_dataset_for_nan_inf(combined_tokenized_train_data, 'input_ids', 'labels')
# check_dataset_for_nan_inf(combined_tokenized_eval_data, 'input_ids', 'labels')
#It doesn't

#%%
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Setup EarlyStoppingCallback if enabled in the configuration
callbacks = []
if config.get('early_stopping', False):  # Default to False if not specified
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.get('early_stopping_patience', 3)))

training_args = Seq2SeqTrainingArguments(
    output_dir=config['output_dir'],
    evaluation_strategy="epoch",  # Choose "epoch" or "steps"
    save_strategy="epoch",        # Make sure this matches evaluation_strategy
    learning_rate=float(config['learning_rate']),
    per_device_train_batch_size=config['per_device_train_batch_size'],
    per_device_eval_batch_size=config['per_device_eval_batch_size'],
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=config['num_train_epochs'],
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    greater_is_better=False,
    # logging_dir=os.path.join(config['output_dir'], "logs"),  # Specify logging directory
    # logging_strategy="epoch"  # Log metrics at each epoch
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=combined_tokenized_train_data,
    eval_dataset=combined_tokenized_eval_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=callbacks
)

trainer.train()

checkpoint_path = os.path.join(config['output_dir'], 'last_model')
# Save the model
model.save_pretrained(checkpoint_path)
# Save the tokenizer
tokenizer.save_pretrained(checkpoint_path)


#%% test on the test set
# test(model_name, config['base_dir'], config['source_lang'], config['target_lang'],generate_prompt)

# %%
