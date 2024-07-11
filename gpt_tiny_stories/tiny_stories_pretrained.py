import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# torch.manual_seed(42)

# Load pretrained model for experimentation
"""
{
  "_name_or_path": "EleutherAI/gpt-neo-125M",
  "activation_function": "gelu_new",
  "attention_layers": [
    "global",
    "local",
    "global",
    "local"
  ],
  "hidden_size": 768,
  "layer_norm_epsilon": 1e-05,
  "max_position_embeddings": 2048,
  "model_type": "gpt_neo",
  "num_heads": 16,
  "num_layers": 4,
  "vocab_size": 50257,
  "window_size": 256
}"""
model_name = 'roneneldan/TinyStories-Instruct-33M'
tokenizer = AutoTokenizer.from_pretrained(model_name)
pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)
pretrained_model.eval()

# Generate completions
prompt = "The cat sat on the mat and wondered"
# prompt = "Lily wanted to get a cat or a dog.  Her mom said that she could not have a cat, so she"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = pretrained_model.generate(input_ids, max_length=200) # greedy
# output = pretrained_model.generate(input_ids, max_length=200, do_sample=True, top_k=50, top_p=0.95, temperature=1.0)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output[0], skip_special_tokens=True))

