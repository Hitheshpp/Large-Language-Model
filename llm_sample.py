from transformers import pipeline

generator = pipeline("text-generation",model = "gpt2")

output = generator("Once upon a time", max_length = 100, num_return_sequences=1)

print(output)