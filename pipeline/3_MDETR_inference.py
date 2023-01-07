import argparse
import pickle
import re
from collections import defaultdict
from os.path import join
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

torch.set_grad_enabled(False);

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def inference(row, glip_demo, transform, args):
    if row['filename'] is None:
      return []
    filename = join(args.image_directory, row['filename'])

    # if no image is found, skip
    if not os.path.exists(filename):
        return []
        
    PIL_image = Image.open(filename).convert('RGB')    
    expression = row[args.expression_column]
    if len(expression) == 0:
        return []
    # mean-std normalize the input image (batch-size: 1)
    img = transform(PIL_image).unsqueeze(0).cuda()

    # propagate through the model
    memory_cache = glip_demo(img, [expression], encode_and_save=True)
    outputs = glip_demo(img, [expression], encode_and_save=False, memory_cache=memory_cache)

    # keep only predictions with 0.7+ confidence
    probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
    keep = (probas > 0.7).cpu()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], PIL_image.size)

    # Extract the text spans predicted by each box
    positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
    predicted_spans = defaultdict(str)

    caption_word = re.findall('\w+|[^\sa-zA-Z0-9]+',expression)
    span = dict()
    for tok in positive_tokens:
        item, pos = tok
        if pos < 255:
            if item in span.keys():
                span[item].append(memory_cache["tokenized"].token_to_word(0, pos))
            else:
                span[item] = [memory_cache["tokenized"].token_to_word(0, pos)]
    for elem in span:
        for span_id in list(set(span[elem])):
            if span_id is not None:
                predicted_spans [elem] += " " + caption_word[min(span_id, len(caption_word) - 1)]


    labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]
    return (probas[keep], bboxes_scaled, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to the input file")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument("--expression_column", type=str, help="Column to translate")
    parser.add_argument("--inference_column", type=str, help="Column to store the inference")
    parser.add_argument("--image_directory", type=str, help="Path to the image directory")
    args = parser.parse_args()

    model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True, return_postprocessor=True)
    model = model.cuda()
    model.eval()

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    tqdm.pandas()

    # Load the data
    data = pickle.load(open(args.input_file, 'rb'))

    # Inference
    data[args.inference_column] = data.progress_apply(lambda row: inference(row, model, transform, args), axis=1)

    # Save the data
    pickle.dump(data, open(args.output_file, 'wb'))
