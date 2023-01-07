# Description: Translate the data using a pretrained model
# Author: Théo Gieruc
# Date: 2022-01-06




import argparse
import pickle

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def translate(sentence, model, tokenizer, args):
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(args.device)
    outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

if __name__ == "__main__":
    tqdm.pandas()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to the input file")
    parser.add_argument("--column", type=str, help="Column to translate")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument("--device", type=str, help="Device to use (cuda or cpu)")
    args = parser.parse_args()


    # check that the column to translate is specified
    assert args.column is not None, "Please specify the column to translate"
    # check that the output is specified
    assert args.output_file is not None, "Please specify the output file"
    # if no args.device is specified, use cuda if available
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available"

    # load the model and the tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en").to(args.device)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

    # load the data using pickle
    with open(args.input_file, "rb") as f:
        data = pickle.load(f)
    # check that the column to translate exists
    assert args.column in data.columns, f"Can't find the column {args.column}"

    # translate the data
    data[f'{args.column}_en'] = data[args.column].progress_apply(lambda x: translate(x, model, tokenizer, args) if pd.notna(x) else '')

    # save the data as a pickle file
    with open(args.output_file, "wb") as f:
        pickle.dump(data, f)

