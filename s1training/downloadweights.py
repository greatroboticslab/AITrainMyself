from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "simplescaling/s1.1-32B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

print("Model downloaded successfully!")






