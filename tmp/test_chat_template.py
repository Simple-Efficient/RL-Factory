from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/share/jjw/huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct")
with open("/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/724/RL-Factory/envs/configs/chat_template.jinja", "r") as f:
    chat_template = f.read()
tokenizer.chat_template = chat_template
message = [
    {"role": "user", "content": "Hello, how are you?"}
]
print(tokenizer.apply_chat_template(message, tools=["sasa"],tokenize=False))