import os
from typing import Any
from datasets import load_dataset, Dataset
import torch
from tqdm import tqdm
import random
from torch.utils.data import Dataset as TorchDataset

class TranslationDataset(TorchDataset):
    def __init__(self, data, tokenizer):
        super(TranslationDataset, self).__init__()

        self.tokenizer = tokenizer
        print(f"Got {len(data)} examples, preprocess...")
        data_dict = self.preprocess(data)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
    def preprocess(self, examples):
        """
        Preprocess the data by tokenizing.
        """
        all_input_ids = []

        print("Tokenizing dataset...")
        for ex in tqdm(examples):
            # Add a positive example
            text = f"{ex['source_text']}\n\nQ: Translate {ex['source_lang']} to {ex['target_lang']}\nA: {ex['target_text']}\n"
            tokenized = self.tokenizer.encode(text)
            all_input_ids.append(torch.LongTensor(tokenized))
        

        random.shuffle(all_input_ids)

        return dict(input_ids=all_input_ids, labels=all_input_ids)
    


def load_and_process_data(base_dir, 
                          train_direction, 
                          target_langs=None):
    train_data = []
    eval_data = []

    # Function to check if a directory matches any language in target_langs
    def is_lang_match(dir_name, langs):
        return any(lang in dir_name for lang in langs)

    # Handle target_langs as a list or a single string
    if target_langs is not None:
        if isinstance(target_langs, str):
            target_langs = [target_langs]
        dirs_with_string = [d for d in os.listdir(base_dir)
                            if os.path.isdir(os.path.join(base_dir, d)) and is_lang_match(d, target_langs)]
    else:
        dirs_with_string = os.listdir(base_dir)

    for dir in dirs_with_string:
        if dir.endswith("-spanish"):
            lang = dir.replace("-spanish", "")
            directions = []
            if train_direction == 'both':
                directions = [(lang, "spanish"), ("spanish", lang)]
            elif train_direction == 'spanish2langs':
                directions = [("spanish", lang)]
            elif train_direction == 'langs2spanish':
                directions = [(lang, "spanish")]

            for source_lang, target_lang in directions:
                # Process training data
                train_file_source = os.path.join(base_dir, dir, f"train.{source_lang}")
                train_file_target = os.path.join(base_dir, dir, f"train.{target_lang}")
                with open(train_file_source, encoding='utf-8') as f1, open(train_file_target, encoding='utf-8') as f2:
                    for source_text, target_text in zip(f1, f2):
                        train_data.append({
                            'source_lang': source_lang,
                            'target_lang': target_lang,
                            'source_text': source_text.strip(),
                            'target_text': target_text.strip()
                        })

                # Process evaluation data
                eval_file_source = os.path.join(base_dir, dir, f"dev.{source_lang}")
                eval_file_target = os.path.join(base_dir, dir, f"dev.{target_lang}")
                with open(eval_file_source, encoding='utf-8') as f1, open(eval_file_target, encoding='utf-8') as f2:
                    for source_text, target_text in zip(f1, f2):
                        eval_data.append({
                            'source_lang': source_lang,
                            'target_lang': target_lang,
                            'source_text': source_text.strip(),
                            'target_text': target_text.strip()
                        })

    return train_data, eval_data

def load_datasets_per_lang(base_dir, 
                          train_direction, 
                          tokenizer,
                          target_langs=None,
                          split='train'):
    

    datasets = {}
    # Function to check if a directory matches any language in target_langs
    def is_lang_match(dir_name, langs):
        return any(lang in dir_name for lang in langs)

    # Handle target_langs as a list or a single string
    if target_langs is not None:
        if isinstance(target_langs, str):
            target_langs = [target_langs]
        dirs_with_string = [d for d in os.listdir(base_dir)
                            if os.path.isdir(os.path.join(base_dir, d)) and is_lang_match(d, target_langs)]
    else:
        dirs_with_string = os.listdir(base_dir)

    for dir in dirs_with_string:
        if dir.endswith("-spanish"):
            lang = dir.replace("-spanish", "")
            directions = []
            if train_direction == 'both':
                directions = [(lang, "spanish"), ("spanish", lang)]
            elif train_direction == 'spanish2langs':
                directions = [("spanish", lang)]
            elif train_direction == 'langs2spanish':
                directions = [(lang, "spanish")]
            
            data = []
            for source_lang, target_lang in directions:
                # Process training data
                file_source = os.path.join(base_dir, dir, f"{split}.{source_lang}")
                file_target = os.path.join(base_dir, dir, f"{split}.{target_lang}")
                with open(file_source, encoding='utf-8') as f1, open(file_target, encoding='utf-8') as f2:
                    for source_text, target_text in zip(f1, f2):
                        data.append({
                            'source_lang': source_lang,
                            'target_lang': target_lang,
                            'source_text': source_text.strip(),
                            'target_text': target_text.strip()
                        })
                        
                datasets[f"{source_lang}-{target_lang}"] = TranslationDataset(
                                                            data=data,tokenizer=tokenizer)

    return datasets



def load_tokenized_data(base_dir, target_langs, train_direction,  preprocess_function, transform_dataset):
    tokenized_train_data = []
    tokenized_eval_data = []

    def is_lang_match(dir_name, langs):
        return any(lang in dir_name for lang in langs)

    if target_langs is not None:
        if isinstance(target_langs, str):
            target_langs = [target_langs]
        dirs_with_string = [d for d in os.listdir(base_dir)
                            if os.path.isdir(os.path.join(base_dir, d)) and is_lang_match(d, target_langs)]
    else:
        dirs_with_string = os.listdir(base_dir)
    
    for dir in dirs_with_string:
        if dir.endswith("-spanish"):
            lang = dir.replace("-spanish", "")
            directions = []
            if train_direction == 'both':
                directions = [(lang, "spanish"), ("spanish", lang)]
            elif train_direction == 'spanish2langs':
                directions = [("spanish", lang)]
            elif train_direction == 'langs2spanish':
                directions = [(lang, "spanish")]

            for source, target in directions:
                print(f"Direction {source} to {target}")
                # Process and tokenize training and evaluation data
                for split in ['train', 'dev']:
                    file_path = os.path.join(base_dir, dir, f"{split}.{source}")
                    with open(file_path, encoding='utf-8') as f1, open(file_path.replace(f"{split}.{source}", f"{split}.{target}"), encoding='utf-8') as f2:
                        translation = [{source: line1.strip(), target: line2.strip()} for line1, line2 in zip(f1, f2)]
                    idx = list(range(len(translation)))
                    books = Dataset.from_dict({"id": idx, "translation": translation})
                    tokenized_dataset = books.map(lambda examples: preprocess_function(examples, source, target), batched=True)
                    transformed_dataset = transform_dataset(tokenized_dataset)

                    if split == 'train':
                        tokenized_train_data.append(transformed_dataset)
                    else:
                        tokenized_eval_data.append(transformed_dataset)

    return tokenized_train_data, tokenized_eval_data
    
def load_tokenized_data_per_lang(base_dir, 
                                 target_langs, 
                                 train_direction, 
                                 preprocess_function, 
                                 transform_dataset,
                                 split='train'):
    tokenized_data_per_lang_pair = {}

    def is_lang_match(dir_name, langs):
        return any(lang in dir_name for lang in langs)

    if target_langs is not None:
        if isinstance(target_langs, str):
            target_langs = [target_langs]
        dirs_with_string = [d for d in os.listdir(base_dir)
                            if os.path.isdir(os.path.join(base_dir, d)) and is_lang_match(d, target_langs)]
    else:
        dirs_with_string = os.listdir(base_dir)

    for dir in dirs_with_string:
        if dir.endswith("-spanish"):
            lang = dir.replace("-spanish", "")
            directions = []
            if train_direction == 'both':
                directions = [(lang, "spanish"), ("spanish", lang)]
            elif train_direction == 'spanish2langs':
                directions = [("spanish", lang)]
            elif train_direction == 'langs2spanish':
                directions = [(lang, "spanish")]

            for source, target in directions:
                print(f"Direction {source} to {target}")
                tokenized_data_for_pair = []  # Store tokenized data for this language pair

                # for split in ['train', 'dev']:
                file_path = os.path.join(base_dir, dir, f"{split}.{source}")
                with open(file_path, encoding='utf-8') as f1, open(file_path.replace(f"{split}.{source}", f"{split}.{target}"), encoding='utf-8') as f2:
                    translation = [{source: line1.strip(), target: line2.strip()} for line1, line2 in zip(f1, f2)]
                idx = list(range(len(translation)))
                books = Dataset.from_dict({"id": idx, "translation": translation})
                tokenized_dataset = books.map(lambda examples: preprocess_function(examples, source, target), batched=True)
                transformed_dataset = transform_dataset(tokenized_dataset)

                # Store the separate tokenized datasets for each language pair
                tokenized_data_per_lang_pair[f"{source}-{target}"] = transformed_dataset

    return tokenized_data_per_lang_pair


class BalancedMultiDataset(TorchDataset):
    def __init__(self, datasets):
        self.datasets = datasets
        # Calculate the total size as the sum of the lengths of all datasets
        self.total_size = sum(len(ds) for ds in datasets.values())

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Choose a random dataset (language pair)
        dataset_name = random.choice(list(self.datasets.keys()))
        dataset = self.datasets[dataset_name]

        # Choose a random example from this dataset
        inner_idx = random.randint(0, len(dataset) - 1)
        return dataset[inner_idx]

    def __str__(self):
        description = f"BalancedMultiDataset containing {len(self.datasets)} language pairs\n"
        for name, ds in self.datasets.items():
            description += f" - {name}: {len(ds)} examples\n"
        return description


class MambaBalancedMultiDataset(TorchDataset):
    def __init__(self, datasets):
        self.datasets = datasets
        # Calculate the total size as the sum of the lengths of all datasets
        self.total_size = sum(len(ds) for ds in datasets.values())

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Choose a random dataset (language pair)
        dataset_name = random.choice(list(self.datasets.keys()))
        dataset = self.datasets[dataset_name]

        # Choose a random example from this dataset
        inner_idx = random.randint(0, len(dataset) - 1)
        # return dataset[inner_idx]
        return dataset[inner_idx]

    def __str__(self):
        description = f"BalancedMultiDataset containing {len(self.datasets)} language pairs\n"
        for name, ds in self.datasets.items():
            description += f" - {name}: {len(ds)} examples\n"
        return description

