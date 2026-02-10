import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL_ID = "liquidai/LFM2-1.2B"
ADAPTER_PATH = "output/EventModel-1.2B"

def load_model():
    print(f"Loading base model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    return model, tokenizer

def generate(model, tokenizer, character_input, max_new_tokens=512):
    prompt = f"### Character: {character_input}\n\n### Problem:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the generated problem part
    if "### Problem:" in generated:
        return generated.split("### Problem:")[-1].strip()
    return generated

def main():
    model, tokenizer = load_model()
    print("\n" + "="*60)
    print("Interactive Inference - Enter character descriptions")
    print("Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("Character: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not user_input:
                continue
            
            print("\nGenerating problem extraction...\n")
            output = generate(model, tokenizer, user_input)
            print(f"Generated Problem:\n{output}\n")
            print("-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
