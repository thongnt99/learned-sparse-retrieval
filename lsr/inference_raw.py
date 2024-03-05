from pathlib import Path
import argparse
from lsr.models import *
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import torch
import json
parser = argparse.ArgumentParser(description="Parsing arguments")
parser.add_argument("--model", type=str)
parser.add_argument("--input", type=str)
parser.add_argument("--max_len", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--write_batch_size", type=int, default=100000)
parser.add_argument("--output", type=str)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--tokenizer", type=str, default="distilbert-base-uncased")

args = parser.parse_args()


def write_to_file(f, batch_results):
    for result in batch_results:
        f.write(json.dumps(result) + "\n")


def encode(model, text_ids, texts, tokenizer, batch_size, output_file_path, max_length=512, write_batch_size=100000, device="cuda", fp16=True):
    f = open(output_file_path, "w")
    res = []
    for idx in range(0, len(text_ids),  batch_size):
        batch_ids = text_ids[idx: (idx+batch_size)]
        batch_texts = texts[idx: (idx+batch_size)]
        batch_inps = tokenizer(
            batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt", return_special_tokens_mask=True).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=fp16):
            reps = model(**batch_inps)
            batch_tok_ids, batch_tok_weights, _ = reps.to_sparse()
            batch_tok_ids = batch_tok_ids.to("cpu").tolist()
            batch_tok_weights = batch_tok_weights.to("cpu").tolist()
            for text_id, tok_ids, tok_weights in zip(batch_ids, batch_tok_ids, batch_tok_weights):
                toks = tokenizer.convert_ids_to_tokens(tok_ids)
                vector = {t: w for t, w in zip(toks, tok_weights) if w > 0}
                json_data = {"id": text_id, "vector": vector}
                res.append(json_data)
                if len(res) == write_batch_size:
                    write_to_file(f, res)
                    res = []
    write_to_file(f, res)
    f.close()


if __name__ == "__main__":
    model = AutoModel.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    input_text_ids = []
    input_texts = []
    with open(args.input, "r") as f:
        for line in tqdm(f, desc=f"Reading input from {args.input}"):
            text_id, text = line.strip().split("\t")
            input_text_ids.append(text_id)
            input_texts.append(text)
    encode(model, text_ids=input_text_ids,
           texts=input_texts, tokenizer=tokenizer, batch_size=args.batch_size, output_file_path= args.output,  max_length=args.max_len, write_batch_size=args.write_batch_size, device=args.device)
