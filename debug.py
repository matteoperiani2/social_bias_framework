# %%
import transformers
from src.model import *

config = transformers.GPT2Config.from_pretrained("distilgpt2")
config.head_dropout = 0.1
model = GPT2WithClassificationHead(config=config)
tokenizer = transformers.AutoTokenizer.from_pretrained("distilgpt2")

tokenizer.pad_token = tokenizer.eos_token
post = ["Niggas are all monkey", "The sky is blue today morning"]
encoded_post =  [tokenizer.encode(s, padding=True, max_length=32) for s in post]

input_ids = torch.tensor(encoded_post)  # Batch size: 1, number of choices: 2

outputs = model(input_ids)
print("end")
