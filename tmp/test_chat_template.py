from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
with open("envs/configs/chat_template.jinja", "r") as f:
    chat_template = f.read()
tokenizer.chat_template = chat_template
message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
]
print(tokenizer.apply_chat_template(message, tokenize=False))