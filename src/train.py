import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import pandas as pd
from datasets import Dataset

MODEL_NAME = "WizardLM/WizardMath-7B-V1.1"

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return Dataset.from_pandas(df)

def tokenize_dataset(dataset, tokenizer):
    return dataset.map(lambda x: tokenizer(x["text"]), batched=True)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto"
    )

    dataset = load_data("data/train.csv")
    dataset = tokenize_dataset(dataset, tokenizer)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs"
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    model.config.use_cache = False
    trainer.train()

    model.save_pretrained("models/lora-math")

if __name__ == "__main__":
    main()