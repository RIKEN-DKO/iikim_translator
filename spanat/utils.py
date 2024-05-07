import os
from tqdm import tqdm
import sacrebleu
from nltk.translate import chrf_score
import torch


def find_checkpoint_directory(base_path):
    for dir_name in os.listdir(base_path):
        if "checkpoint-" in dir_name:
            return os.path.join(base_path, dir_name)
    raise FileNotFoundError("No checkpoint directory found in the given model path.")


def clean_mamba_output(decoded_text,original_text):
    cleaned = decoded_text.replace(original_text, "")
    cleaned = cleaned.replace("<|endoftext|>", "")
    cleaned = cleaned.replace("\n", "")
    
    return cleaned

def batched_inference_mamba(
    texts, 
    tokenizer, 
    model, 
    prompt, 
    batch_size=8, 
    max_length=512, 
    max_new_tokens=256
):
    results = []
    iterable = range(0, len(texts), batch_size)
    iterable = tqdm(iterable, total=len(iterable), desc="Processing batches")
    generate_prompt = prompt
    for i in iterable:
        batch_questions = [generate_prompt(q) for q in texts[i : i + batch_size]]
        batch = tokenizer(
            batch_questions, return_tensors="pt", padding=True, truncation=True
        )
        # with torch.cuda.amp.autocast():
        #     # output_tokens = model.generate(input_ids=batch["input_ids"].cuda(), max_new_tokens=max_new_tokens, do_sample=False)
        #     #MAMBA_T5 
        # Ensure input_ids are of type torch.LongTensor
        batch_input_ids = batch["input_ids"].long().cuda()

        output_tokens = model.generate(
            input_ids=batch_input_ids,
            # max_new_tokens=max_new_tokens,
            max_length=max_length,
            eos_token_id=tokenizer.eos_token_id,
        )
        # batch_out = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in output_tokens]
        batch_out = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        #Get only the answers
        clean_batch = []
        for i,answer in enumerate(batch_out):
            clean_batch.append(clean_mamba_output(answer,batch_questions[i]))
        
        results.extend(clean_batch)

    return results


def batched_inference(
    texts, 
    tokenizer, 
    model, 
    prompt, 
    batch_size=8, 
    max_length=512, 
    max_new_tokens=256
):
    results = []
    iterable = range(0, len(texts), batch_size)
    iterable = tqdm(iterable, total=len(iterable), desc="Processing batches")
    generate_prompt = prompt
    for i in iterable:
        batch_questions = [generate_prompt(q) for q in texts[i : i + batch_size]]
        batch = tokenizer(
            batch_questions, return_tensors="pt", padding=True, truncation=True
        )
        with torch.cuda.amp.autocast():
            # output_tokens = model.generate(input_ids=batch["input_ids"].cuda(), max_new_tokens=max_new_tokens, do_sample=False)
            #MAMBA_T5 
            output_tokens = model.generate(
                input_ids=batch["input_ids"].cuda(),
                # max_new_tokens=max_new_tokens,
                max_length=max_length,
                eos_token_id=tokenizer.eos_token_id,
            )
        # batch_out = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in output_tokens]
        batch_out = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        #Get only the answers
        clean_batch = []
        for i,answer in enumerate(batch_out):
            clean_batch.append(clean_mamba_output(answer,batch_questions[i]))
        
        results.extend(clean_batch)

    return results

def batched_inference_m2m(
    texts, 
    tokenizer, 
    model, 
    prompt, 
    batch_size=8, 
    max_length=512, 
    max_new_tokens=256
):
    results = []
    iterable = range(0, len(texts), batch_size)
    iterable = tqdm(iterable, total=len(iterable), desc="Processing batches")
    generate_prompt = prompt
    for i in iterable:
        batch_questions = [generate_prompt(q) for q in texts[i : i + batch_size]]
        batch = tokenizer(
            batch_questions, return_tensors="pt", padding=True, truncation=True
        )
        
        input_ids = batch["input_ids"].to("cuda:0")  # You can specify any GPU here as the starting point
        model.eval()
        # print(input_ids.shape)
        with torch.cuda.amp.autocast():
            # output_tokens = model.generate(input_ids=batch["input_ids"].cuda(), max_new_tokens=max_new_tokens, do_sample=False)
        # Ensure your batch is on the correct device
            
            output_tokens = model.module.generate(  # Use 'model.module' to access the underlying model when using DataParallel
                input_ids=input_ids,
                max_length=max_length,
                eos_token_id=tokenizer.eos_token_id,
            )
        # batch_out = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in output_tokens]
        batch_out = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        #Get only the answers
        clean_batch = []
        for i,answer in enumerate(batch_out):
            clean_batch.append(clean_mamba_output(answer,batch_questions[i]))
        
        results.extend(clean_batch)

    return results

from tqdm import tqdm

def batched_inference_m2m_accelerate(
    accelerator,
    texts, 
    tokenizer, 
    model, 
    prompt, 
    batch_size=8, 
    max_length=512, 
    max_new_tokens=256
):
    results = []
    iterable = range(0, len(texts), batch_size)
    iterable = tqdm(iterable, total=len(iterable), desc="Processing batches")
    generate_prompt = prompt
    for i in iterable:
        batch_questions = [generate_prompt(q) for q in texts[i : i + batch_size]]
        batch = tokenizer(
            batch_questions, return_tensors="pt", padding=True, truncation=True
        )

        # Prepare your batch for the appropriate device
        batch = accelerator.prepare(batch)
                # Ensure batch is on the correct device
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        # print(batch["input_ids"].shape)
        # model.eval()
        with torch.inference_mode():
            actual_model = model.module if hasattr(model, 'module') else model
            output_tokens = actual_model.generate(
                input_ids=batch["input_ids"],
                max_length=max_length,
                eos_token_id=tokenizer.eos_token_id,
            )
        batch_out = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        # Your cleaning and result preparation code
        clean_batch = [clean_mamba_output(answer, batch_questions[i]) for i, answer in enumerate(batch_out)]
        
        results.extend(clean_batch)

    return results



def calculate_score_report(sys, ref, score_only):
    # chrf = chrf_score.corpus_chrf(sys, ref)
    # this is chrf++
    chrf = sacrebleu.corpus_chrf(sys, ref, word_order=2)
    chrf = round(chrf.score, 2)
    bleu = sacrebleu.corpus_bleu(sys, ref)
    print("chrf:", chrf)
    print("BLEU:", bleu.format(score_only=score_only))
    return "chrf: " + str(chrf) + "\nBLEU: " + str(bleu.format(score_only=score_only))


def append_results(
    output_dir, subdir, file_type, score_report, results, gold_lines, append_mode="a"
):
    file_dir = os.path.join(output_dir, file_type)
    os.makedirs(file_dir, exist_ok=True)

    with open(
        os.path.join(file_dir, f"{file_type}.txt"), append_mode, encoding="utf-8"
    ) as f:
        f.write(f"\nResults for {file_type} from {subdir}:\n")
        f.write(score_report)

    with open(
        os.path.join(file_dir, f"results_{file_type}.txt"),
        append_mode,
        encoding="utf-8",
    ) as f:
        f.write(f"\nResults for {file_type} from {subdir}:\n")
        f.write("\n".join(results))

    with open(
        os.path.join(file_dir, f"gold_lines_{file_type}.txt"),
        append_mode,
        encoding="utf-8",
    ) as f:
        f.write(f"\nGold lines for {file_type} from {subdir}:\n")
        f.write("\n".join(gold_lines))


def check_and_get_file_paths(base_data_dir, source_lang, target_lang, file_type):
    sub_dir = f"{target_lang}-{source_lang}"
    source_file = os.path.join(base_data_dir, sub_dir, f"{file_type}.{source_lang}")
    target_file = os.path.join(base_data_dir, sub_dir, f"{file_type}.{target_lang}")

    if not os.path.exists(source_file) or not os.path.exists(target_file):
        sub_dir = f"{source_lang}-{target_lang}"
        source_file = os.path.join(base_data_dir, sub_dir, f"{file_type}.{source_lang}")
        target_file = os.path.join(base_data_dir, sub_dir, f"{file_type}.{target_lang}")

    if os.path.exists(source_file) and os.path.exists(target_file):
        return source_file, target_file
    else:
        return None, None

#Function that save a list of lines into a files
    
def save_lines_to_file(lines, file_path):
    # Check if the directory of the file_path exists, if not, create it
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Now that we're sure the directory exists, write the lines to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def test(
    model_path,
    base_data_dir,
    source_lang,
    target_lang,
    prompt,
    model,
    tokenizer,
    batch_size=32,
    max_new_tokens=256,
    max_length=512,
    file_types=["test"],
    append_mode="a",
    mode = 'T5',
):
    if mode =='T5':
        inference = batched_inference
    elif mode == 'mamba':
        inference = batched_inference_mamba
    elif mode == 'm2m':
        inference = batched_inference_m2m

        
        
    # model_checkpoint_dir = find_checkpoint_directory(model_path)

    for file_type in file_types:
        source_file, target_file = check_and_get_file_paths(
            base_data_dir, source_lang, target_lang, file_type
        )
        print("<<  ",file_type,"  >>")
        if source_file: #and target_file:
            with open(source_file, encoding="utf-8") as f:
                source_lines = [line.strip() for line in f.readlines()]

            examples = [
                {
                    "source_lang": source_lang,
                    "source_text": line,
                    "target_lang": target_lang,
                }
                for line in source_lines
            ]
            
            results = inference(
                texts=examples,
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
            )       
            if target_lang == "chatino":
                #Add the superscripts
                results = modify_4chatino(results)

            save_lines_to_file(results,os.path.join(
                model_path,file_type, f"{target_lang}_results.txt"))
            
            if target_file:
                with open(target_file, encoding="utf-8") as f:
                    gold_lines = [line.strip() for line in f.readlines()]

                assert len(results) == len(gold_lines)
                

                score_report = calculate_score_report(results, [gold_lines], score_only=True)

                append_results(
                    model_path,
                    target_lang + "-" + source_lang,
                    file_type,
                    score_report,
                    results,
                    gold_lines,
                    append_mode=append_mode,
                )



def test_accelerator(
    model_path,
    base_data_dir,
    source_lang,
    target_lang,
    prompt,
    model,
    tokenizer,
    batch_size=32,
    max_new_tokens=256,
    max_length=512,
    file_types=["test"],
    append_mode="a",
    mode = 'T5',
    accelerator=None,
):
    if mode =='T5':
        inference = batched_inference
    elif mode == 'mamba':
        inference = batched_inference_mamba
    elif mode == 'm2m':
        inference = batched_inference_m2m
    elif mode == 'm2m_accelerate':
        inference = batched_inference_m2m_accelerate
        
        
    # model_checkpoint_dir = find_checkpoint_directory(model_path)

    for file_type in file_types:
        source_file, target_file = check_and_get_file_paths(
            base_data_dir, source_lang, target_lang, file_type
        )
        print("<<  ",file_type,"  >>")
        if source_file and target_file:
            with open(source_file, encoding="utf-8") as f:
                source_lines = [line.strip() for line in f.readlines()]

            examples = [
                {
                    "source_lang": source_lang,
                    "source_text": line,
                    "target_lang": target_lang,
                }
                for line in source_lines
            ]
            
            results = inference(
                texts=examples,
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                accelerator=accelerator
            )     
              
            accelerator.wait_for_everyone()

            if accelerator.is_local_main_process:
                with open(target_file, encoding="utf-8") as f:
                    gold_lines = [line.strip() for line in f.readlines()]

                assert len(results) == len(gold_lines)
                
                save_lines_to_file(results,os.path.join(
                    model_path,file_type, f"{target_lang}_results.txt"))

                score_report = calculate_score_report(results, [gold_lines], score_only=True)

                append_results(
                    model_path,
                    target_lang + "-" + source_lang,
                    file_type,
                    score_report,
                    results,
                    gold_lines,
                    append_mode=append_mode,
                )




import re
def modify_4chatino(strings):
    # Dictionary mapping specified characters to their superscript versions
    superscript_map = {
        'A': 'ᴬ', 'B': 'ᴮ', 'c': 'ᶜ', 'E': 'ᴱ', 'f': 'ᶠ',
        'G': 'ᴳ', 'H': 'ᴴ', 'I': 'ᴵ', 'J': 'ᴶ', 'K': 'ᴷ'
    }
    
    # Function to replace the appropriate character if it is in the specified list
    def replace_superscript(word):
        # Handle words ending with a comma
        if len(word) > 2 and word[-1] == ',' and word[-2] in superscript_map:
            return word[:-2] + superscript_map[word[-2]] + ','
        # Handle words ending with one of the specified characters
        elif len(word) > 2 and word[-1] in superscript_map:
            return word[:-1] + superscript_map[word[-1]]
        return word
    
    # Function to replace 'J' with 'ᴶ' if it is between two lowercase letters
    def replace_j(word):
        return re.sub(r'(?<=[a-z])J(?=[a-z])', 'ᴶ', word)

    modified_strings = []
    for line in strings:
        # Split the line into words, modify them, and join back into a string
        words = line.split()
        words = [replace_j(replace_superscript(word)) for word in words]
        modified_strings.append(' '.join(words))
    
    return modified_strings

