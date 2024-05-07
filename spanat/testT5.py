import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from .utils import find_checkpoint_directory, batched_inference, calculate_score_report
from .utils import append_results,check_and_get_file_paths,test

def generate_prompt(example):
    prompt = f"translate {example['source_lang']} to {example['target_lang']}: {example['source_text']}"
    return prompt

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map='auto')
    return tokenizer, model



def main():
    parser = argparse.ArgumentParser(description="Model metrics measurement script.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--base_data_dir", type=str, required=True, help="Base directory for data.")
    parser.add_argument("--source_lang", type=str, required=True, help="Source language.")
    parser.add_argument("--target_lang", type=str, required=True, help="Target language.")
    parser.add_argument("--append_mode", type=str, required=False, default='w',help="Write or append to the results files")
    parser.add_argument("--from_hf", action='store_true',default=False)
    
    args = parser.parse_args()
    hf=''
    if args.from_hf:
        print("Loading from HF")
        tokenizer, model = load_model(args.model_path)
        hf="_HF"
    else:
        print('Searching checkpoint')
        model_checkpoint_dir = find_checkpoint_directory(args.model_path)    
        tokenizer, model = load_model(model_checkpoint_dir)

    test(model_path=args.model_path+hf,
         base_data_dir=args.base_data_dir, 
         source_lang=args.source_lang, 
         target_lang=args.target_lang, 
         prompt=generate_prompt,
         model=model,
         tokenizer=tokenizer,
         batch_size=256,
         max_new_tokens=256,
         max_length=256,
         file_types=['test','dev'],
         append_mode=args.append_mode,
         mode="T5"
         ) #Rewrite the results

if __name__ == "__main__":
    main()