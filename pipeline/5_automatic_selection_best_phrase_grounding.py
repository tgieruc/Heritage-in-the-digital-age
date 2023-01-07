import pickle
import pandas as pd
import argparse
from tqdm import tqdm
from multiprocess import Pool
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="Path to the input file")
parser.add_argument("--output_file", type=str, help="Path to the output file")
parser.add_argument("--selection_column", type=str, help="Column to store the selection")

args = parser.parse_args()


def select_best_phrase_grounding(row, classifier):
    possibilities = ['MDETR_caption', 'MDETR_title', 'GLIP_caption', 'GLIP_title']

    # exclude empty possibilities
    possibilities = [p for p in possibilities if row[p] is not None and len(row[p]) > 0]
    possibilities = [p for p in possibilities if row[p][0] is not None and len(row[p][0]) > 0]

    if len(possibilities) == 0: # no possibilities
        return None
    if len(possibilities) == 1: # only one possibility
        return possibilities[0]
    if (('MDETR_caption' in possibilities) or ('GLIP_caption' in possibilities)) and (('MDETR_title' in possibilities) and ('GLIP_title' in possibilities)): # at least one caption and one title
        predictions = classifier([row['caption'].lower(), row['title_en'].lower()])
        scores = [p['label'] for p in predictions]
        label = scores.index(max(scores))
        if label == 0:
            if 'GLIP_caption' in possibilities:
                return 'GLIP_caption'
            else:
                return 'MDETR_caption'
        else:
            if 'GLIP_title' in possibilities:
                return 'GLIP_title'
            else:
                return 'MDETR_title'
    if 'GLIP_caption' in possibilities: # GLIP_caption vs MDETR_caption
        return 'GLIP_caption'
    if 'GLIP_title' in possibilities:  # GLIP_title vs MDETR_title
        return 'GLIP_title'



if __name__ == '__main__':

    tqdm.pandas()

    with open(args.input_file, "rb") as f:
        df = pickle.load(f)

    classifier = pipeline(model="tgieruc/Heritage-in-the-Digital-Age-expression-ranking-pipeline", trust_remote_code=True,device=-1)

    dict_of_df = df.to_dict('records')

    with Pool(8) as p:
        df[args.selection_column] = list(tqdm(p.imap(lambda x: select_best_phrase_grounding(x, classifier), dict_of_df), total=len(dict_of_df)))

    with open(args.output_file, "wb") as f:
        pickle.dump(df, f)
        