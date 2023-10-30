# %%
import transformers
from src.model import *
from src.losses import *

config = transformers.GPT2Config.from_pretrained("distilgpt2")
config.head_dropout = 0.1
model = GPT2WithClassificationHead(config=config)
tokenizer = transformers.AutoTokenizer.from_pretrained("distilgpt2")

tokenizer.pad_token = tokenizer.eos_token
post = ["Niggas are all monkey<|endoftext|>", "Niggas are all monkey<|endoftext|>"]
encoded_post =  [tokenizer.encode(s) for s in post]

input_ids = torch.tensor(encoded_post)
outputs = model(input_ids)

c_labels = torch.tensor([[0,1], [0,0],[1,1],[0,1],[0,1]], dtype=float)
weight = torch.tensor([0.44, 0.6, 0.8, 0.34, 0.41])
clssf_loss = classification_loss(outputs.clssf_logits, c_labels, weight)
print("end")
