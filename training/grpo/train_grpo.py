import argparse
import json
import re

from datasets import Dataset

from training.grpo.reward import latency_reward_func


def remove_sql_comments(sql):
    sql = re.sub(r"--.*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = re.sub(r"\n+", "\n", sql, flags=re.DOTALL)
    return sql.strip()


def load_grpo_dataset(path):
    with open(path, "r") as fin:
        raw_data = json.load(fin)
    data_set = {"prompt": [], "answer": []}
    for data in raw_data:
        prompt = "*** sql ***\n{}\n".format(remove_sql_comments(data["sql"]))
        prompt += f"*** table information *** :\n{data['card']}\n"
        data_set["prompt"].append([{"role": "user", "content": prompt}])
        data_set["answer"].append(data["pg_exectime"])
    return Dataset.from_dict(data_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="SFT checkpoint directory")
    parser.add_argument("--data", default="training/grpo/job_grpo.json")
    parser.add_argument("--output_dir", default="grpo-output")
    args = parser.parse_args()

    dataset = load_grpo_dataset(args.data)

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import GRPOConfig, GRPOTrainer

    max_seq_length = 2500
    lora_rank = 32
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.9,
    )
    model = FastLanguageModel.get_peft_model(model, r=lora_rank, lora_alpha=lora_rank)
    model.gradient_checkpointing_enable()

    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=1500,
        max_completion_length=1000,
        max_steps=200,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=args.output_dir,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[latency_reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
