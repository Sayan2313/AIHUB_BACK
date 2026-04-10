from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import lru_cache
from path_resolver import qwen_model_path
@lru_cache
def inference(sentence:str):
    model_path = qwen_model_path
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto"
    )
    messages = [
        {"role": "user", "content": sentence},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        tokenizer(prompt, return_tensors="pt").to("cuda")["input_ids"],
        max_new_tokens=100,
        eos_token_id=[
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.eos_token_id  # fallback
        ]
    )
    input_length = inputs["input_ids"].shape[1]
    output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return output