#
def prompt_openllama(example):
    source_lang = example['source_lang'].capitalize()
    target_lang = example['target_lang'].capitalize()
    prompt_text = f"#{source_lang}:\n{example['source_text']}\n#{target_lang}:"
    return prompt_text

#
def prompt_openllama_train(example):
    text = f"{prompt_openllama(example)}\n{example['target_text']}"
    return text