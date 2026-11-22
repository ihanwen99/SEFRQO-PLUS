MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
LLAMA3_STOP_TOKEN_IDS = [128009, 128001]


def load_vllm(model_path):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(model=model_path, dtype="bfloat16")
    params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        n=1,
        stop_token_ids=LLAMA3_STOP_TOKEN_IDS,
    )

    def generate(messages):
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return llm.generate(prompt, params)[0].outputs[0].text.strip()

    return generate


def load_peft_model(adapter_path, base_model_path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, local_files_only=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    def generate(messages):
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    return generate


def load_hf(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    def generate(messages):
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            outputs[0][inputs.shape[1] :], skip_special_tokens=True
        ).strip()

    return generate


def load_generator(model_cfg):
    loader = model_cfg["loader"]
    if loader == "vllm":
        return load_vllm(model_cfg["path"])
    if loader == "peft":
        return load_peft_model(model_cfg["path"], model_cfg["base_path"])
    if loader == "hf":
        return load_hf(model_cfg["path"])
    raise ValueError(f"Unknown loader: {loader}")
