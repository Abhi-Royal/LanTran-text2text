from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load a pre-trained T5 model for text-to-text tasks
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large",legacy=False)

user_input = input(f"Question: ")
# Tokenized input
input_ids = tokenizer.encode(user_input, return_tensors="pt")

# Generate output
outputs = model.generate(input_ids, max_length=50)

# Decode output tokens
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Answer: ", result)