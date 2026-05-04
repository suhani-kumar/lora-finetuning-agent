import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "WizardLM/WizardMath-7B-V1.1"
ADAPTER_PATH = "models/lora-math"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    return tokenizer, model

def generate(prompt):
    tokenizer, model = load_model()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = """
    Solve:
    In triangle ABC, angle B is 90°, BC = 16, AC = 20.
    What is sin C?
    """

    print(generate(prompt))