import json
import torch
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Configuration
MODEL_ID = "liquidai/LFM2-1.2B"
DATA_PATH = "output/processed.jsonl"
OUTPUT_DIR = "output/EventModel-1.2B"

def prepare_dataset(path):
    """
    Loads processed data and prepares it for SFT.
    """
    examples = []
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None

    print(f"Processing {path}...")
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                
                # Parse model response safely
                if isinstance(item.get('model_response'), str):
                    resp = json.loads(item['model_response'])
                else:
                    resp = item.get('model_response', {})

                chars = resp.get('characters', [])
                
                if not chars:
                    continue
                    
                # Handle both flat lists ["A", "B"] and nested lists [["A"], ["B"]]
                if isinstance(chars[0], str):
                    trait_sets = [chars] 
                else:
                    trait_sets = chars

                for traits in trait_sets:
                    char_str = ", ".join([str(t) for t in traits])
                    problem = resp.get('problem', 'No problem identified.')
                    text = f"### Character: {char_str}\n\n### Problem: {problem}"
                    examples.append({"text": text})

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_num}")
            except Exception as e:
                continue
    
    print(f"Prepared {len(examples)} training examples.")
    return Dataset.from_list(examples)

def train():
    # 1. Dataset Preparation
    dataset = prepare_dataset(DATA_PATH)
    if dataset is None or len(dataset) == 0:
        print("No data found or dataset is empty. Exiting.")
        return

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Fix padding for SFT
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 2. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear", 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 3. Training Arguments (SFTConfig)
    # UPDATED: max_length and dataset_text_field are now here
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=1024,            # Renamed from max_seq_length
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=3,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        packing=False
    )

    # 4. Initialize Trainer
    # UPDATED: 'tokenizer' renamed to 'processing_class'
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer, # Fixes "unexpected keyword argument 'tokenizer'"
        args=training_args,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Finetuning complete.")

if __name__ == "__main__":
    train()