from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel,GPT2Tokenizer,DataCollatorForLanguageModeling
import torch
from datasets import load_dataset
from transformers import Trainer,TrainingArguments
TRAIN_BASE= False
paths = ["python_code_text_data.txt"]
if TRAIN_BASE:
    tokenizer = ByteLevelBPETokenizer()   

    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2,special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    tokenizer.save_model("tokenizer")


inp = 'print("Hello world!")'
tokenizer = GPT2Tokenizer.from_pretrained("tokenizer")

tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>"
})
t= tokenizer.encode(inp)
print(t)
print(tokenizer.decode(t))

model = GPT2LMHeadModel.from_pretrained("GPyT").to("cpu")
while True:
    inp = input(">>> ")
    input_ids = tokenizer.encode(inp,return_tensors="pt").to("cpu")
    
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == tokenizer.pad_token_id] = 0

    beam_output = model.generate(
        input_ids,
        max_length = 512,
        num_beams = 10,
        temperature=0.7,
        do_sample=True,
        no_repeat_ngram_size=5,
        attention_mask=attention_mask,
        num_return_sequences=1)
    for beam in beam_output:
        out = tokenizer.decode(beam)
        fout=out.replace("<N>","\n")
        print(str(fout))