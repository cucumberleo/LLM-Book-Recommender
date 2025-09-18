# fine-tune-Qwen3-8B-Qlora-fixed.py
import json
import torch
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import gc
import torch

# 准备数据集
def load_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            formatted_text = f"### Instruction: Generate a book recommendation response\n### Input: {item['prompt']}\n### Response: {item['completion']}"
            data.append({"text": formatted_text})
    return Dataset.from_list(data)

# QLoRA config
def setup_qlora(model_name):
    # 4 bit 量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ensure pad token defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    return model, tokenizer, peft_config


# fine tune Qwen3-8B
def fine_tune_Qwen():
    gc.collect()
    torch.cuda.empty_cache()
    model_name = "Qwen/Qwen3-8B"
    model, tokenizer, peft_config = setup_qlora(model_name)

    dataset = load_dataset("synthetic_book_recommendations.jsonl")

    training_arguments = SFTConfig(
        output_dir="./Qwen3-8B-book-recommender",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=4,
        fp16=True,
        push_to_hub=False,
        report_to=None,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        dataset_text_field="text",
        max_length=384,   
        packing=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False
    )
    def formatting_func(example):
        return example["text"]

    # 训练
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        args=training_arguments,
        # resume_from_checkpoint="./Qwen3-8B-book-recommender/checkpoint-45"  fix:参数
    )
    trainer.train(resume_from_checkpoint="./Qwen3-8B-book-recommender/checkpoint-45")
    # 保存模型
    trainer.model.save_pretrained("./Qwen3-8B-book-recommender-final")
    tokenizer.save_pretrained("./Qwen3-8B-book-recommender-final")
    # 删除缓存
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    fine_tune_Qwen()
