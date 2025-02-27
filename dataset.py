import os
import argparse
from time import sleep
import torch
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer


# ------------------------------------------

parser = argparse.ArgumentParser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.add_argument("-t", "--type", type=str, default="classic", help="Fineweb type, edu|classic")
parser.add_argument("-v", "--version", type=str, default="10B", help="Fineweb data sample size, 10B|100B")
parser.add_argument("-n", "--tokenizer", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="HuggingFace tokenizer")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each data shard in the output .pt files, in tokens")
parser.add_argument("-b", "--batch_size", type=int, default=2**16, help="Size of each data shard in the output .pt files, in tokens")
args = parser.parse_args()


# FineWeb has a few possible subsamples available
assert args.version in {"10B", "100B"}, "version must be one of: 10B, 100B"
assert args.type in {"edu", "classic"}, "type must be one of: edu, classic"
directories = {
    ("classic", "10B"):     ("HuggingFaceFW/fineweb",       "10B",      "sample-10BT"),
    ("classic", "100B"):    ("HuggingFaceFW/fineweb",       "100B",     "sample-100BT"),
    ("edu",     "10B"):     ("HuggingFaceFW/fineweb-edu",   "edu_10B",  "sample-10BT"),
    ("edu",     "100B"):    ("HuggingFaceFW/fineweb-edu",   "edu_100B", "sample-100BT")
}
dataset_dir, local_dir, name = directories[(args.type, args.version)]

os.makedirs(f'./data/{local_dir}', exist_ok=True)
# ------------------------------------------

def tokenize_shard(start, end, dataset, tokenizer, dtype):
    tokens = tokenizer(dataset[start:end]['text'], padding=False, truncation=False, return_length=True)
    token_count = sum(tokens['length'])
    all_input_ids = []
    all_seq_codes = []
    for i in range(len(tokens['input_ids'])):
        all_input_ids.extend(tokens['input_ids'][i])
        all_seq_codes.append(torch.full((len(tokens['input_ids'][i]),), start+i, dtype=torch.int32))
    return torch.tensor(all_input_ids, dtype=dtype), torch.concat(all_seq_codes), token_count


def write_datafile(filename, input_ids, seq_codes, tokenizer_name):
    data_dict = {
        "tokenizer": tokenizer_name,
        "input_ids": input_ids,
        "seq_codes": seq_codes
    }
    torch.save(data_dict, filename)

print('Loading dataset')
dataset = load_dataset(dataset_dir, name=name, split="train")

print('Loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, add_eos_token=True)

token_dtype = torch.uint16 if tokenizer.vocab_size < 2**16 else torch.uint32

print('Loading First Shard')
progress_bar = tqdm(total=len(dataset), unit="File", desc=f"Shard {0}")
input_ids, seq_codes, token_count = tokenize_shard(0, args.batch_size, dataset, tokenizer, token_dtype)
progress_bar.update(args.batch_size)

current_pos = args.batch_size
shard_index = 0
token_count = 0
while current_pos < len(dataset):
    if len(input_ids) < args.shard_size:
        new_input_ids, new_seq_codes, current_token_count = tokenize_shard(current_pos, current_pos+args.batch_size, dataset, tokenizer, token_dtype)
        token_count += current_token_count
        input_ids = torch.cat((input_ids, new_input_ids), dim=0)
        seq_codes = torch.cat((seq_codes, new_seq_codes), dim=0)

        if current_pos+args.batch_size < len(dataset):
            progress_bar.update(args.batch_size)
            current_pos += args.batch_size
        else:
            progress_bar.update(len(dataset) - current_pos)
            current_pos = len(dataset)
    else:
        filename = os.path.join(f'./data/{local_dir}', f"sample_{shard_index:06d}.pt")
        write_datafile(filename, input_ids[:args.shard_size], seq_codes[:args.shard_size], args.tokenizer)
        shard_index += 1

        # Update progress bar description
        progress_bar.set_description(f"Shard {shard_index}")

        # populate the next shard with the leftovers of the current doc
        input_ids = input_ids[args.shard_size:]
        seq_codes = seq_codes[args.shard_size:]


if len(input_ids) != 0:
    filename = os.path.join(f'./data/{local_dir}', f"sample_{shard_index:06d}.pt")
    write_datafile(filename, input_ids, seq_codes, args.tokenizer)
    shard_index += 1
    progress_bar.update(len(input_ids))
sleep(10)
print("Done")
print(f"Total shards: {shard_index}")
print(f"Total tokens: {token_count}")



        
    # if token_count + len(tokens) < args.shard_size:
    #     # simply append tokens to current shard
    #     all_tokens_np[token_count:token_count+len(tokens)] = tokens
    #     token_count += len(tokens)
    #     # update progress bar
    #     if progress_bar is None:
    #         progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
    #     progress_bar.update(len(tokens))
    # else:
    #     # write the current shard and start a new one
    #     split = "val" if shard_index == 0 else "train"
    #     filename = os.path.join(DATA_CACHE_DIR, f"sample_{split}_{shard_index:06d}.pt")
    #     # split the document into whatever fits in this shard; the remainder goes to next one
    #     remainder = args.shard_size - token_count
    #     progress_bar.update(remainder)
    #     all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
    #     write_datafile(filename, all_tokens_np.tolist(), args.model_desc)
    #     shard_index += 1
    #     progress_bar = None
    #     # populate the next shard with the leftovers of the current doc
    #     all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
    #     token_count = len(tokens)-remainder
