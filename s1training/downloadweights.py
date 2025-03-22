from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "simplescaling/s1.1-32B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# You can now use the model for inference!
