

def base_trans_prompt(source_lang, target_lang, source_text):
    ex = {'source_lang':source_lang,
        'target_lang':target_lang,
        'source_text':source_text}
    
    return trans_prompt(ex)

def trans_prompt(example):
    prompt = f"translate {example['source_lang']} to {example['target_lang']}: {example['source_text']}"

    return prompt

