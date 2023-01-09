import argparse
import pickle
from tqdm import tqdm
import copy
from torchvision.ops import nms
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="Path to the input file")
parser.add_argument("--output_file", type=str, help="Path to the output file")
parser.add_argument("--columns_to_process",  nargs='+', help="Columns to process", default=['title', 'caption'])

args = parser.parse_args()

def labelwise_nms(segmentation_results, iou_th=0.2):
    if len(segmentation_results) == 0:
        return segmentation_results

    unique_caption = set(segmentation_results[2])
    
    if len(segmentation_results[2]) == 0:
        return segmentation_results

    boolean_index = [[elem_ == cap for elem_ in segmentation_results[2]] for cap in list(unique_caption)]
    idx = [[i for i, x in enumerate(bool_idx) if x] for bool_idx in boolean_index]
    idx_to_keep = [nms(boxes=torch.index_select(segmentation_results[1], 0, torch.tensor(idx_)),
                        scores=torch.index_select(segmentation_results[0], 0, torch.tensor(idx_)), iou_threshold=iou_th) for idx_ in
                    idx]

    # Get scores, boxes, and captions
    scores = []
    boxes = []
    captions = []
    for idx_, idx_tokeep, caption in zip(idx, idx_to_keep, list(unique_caption)):
        scores += (segmentation_results[0][idx_][idx_tokeep])
        boxes += (segmentation_results[1][idx_][idx_tokeep])
        captions += (
            [segmentation_results[2][i] for i in torch.index_select(torch.tensor(idx_), 0, idx_tokeep)])
    captions = [caption[:1] if caption[0] == ' ' else caption for caption in captions]
    # Make the new segmentation
    segmentation_nms = []
    segmentation_nms.append(torch.stack(scores, dim=0))
    segmentation_nms.append(torch.stack(boxes, dim=0))
    segmentation_nms.append(captions)

    return segmentation_nms


def global_nms(segmentation_results, iou_th=0.9):
    if len(segmentation_results) == 0:
        return segmentation_results
    if len(segmentation_results[2]) == 0:
        return segmentation_results

    idx_to_keep = nms(boxes=segmentation_results[1], scores=segmentation_results[0], iou_threshold=iou_th)
    scores = []
    boxes = []
    captions = []
    scores += (segmentation_results[0][idx_to_keep])
    boxes += (segmentation_results[1][idx_to_keep])
    captions += ([segmentation_results[2][i] for i in idx_to_keep])
    captions = [caption[:1] if caption[0] == ' ' else caption for caption in captions]

    # Make the new segmentation
    segmentation_nms = []
    segmentation_nms.append(torch.stack(scores, dim=0))
    segmentation_nms.append(torch.stack(boxes, dim=0))
    segmentation_nms.append(captions)

    return segmentation_nms

if __name__ == '__main__':

    tqdm.pandas()

    with open(args.input_file, 'rb') as f:
        data = pickle.load(f)

    for column in args.columns_to_process:
        data[column] = data[column].progress_apply(lambda x: global_nms(labelwise_nms(x)))

    with open(args.output_file, 'wb') as f:
        pickle.dump(data, f)
