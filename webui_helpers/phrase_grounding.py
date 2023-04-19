import os, sys
from tqdm import tqdm
import torch
from torchvision.ops import box_convert
tqdm.pandas()

from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate

home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def dino_predict(row, model, image_dir, caption_col, device):

    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    filename = os.path.join(image_dir, row['filename'])

    assert os.path.exists(filename), f"Image {filename} does not exist"

    text_prompt = row[caption_col]

    image_source, image = load_image(filename)

    boxes, logits, labels = predict(
        model=model, 
        image=image, 
        caption=text_prompt, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )

    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").int()

    return {'bounding_boxes': xyxy, 'scores': logits, 'labels': labels}

def run_DINO(dataframe, image_directory, caption_columns, device):

    CONFIG_PATH =  os.path.join(home_dir,"submodules/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    WEIGHTS_PATH = os.path.join(home_dir,"submodules/GroundingDINO/model/groundingdino_swint_ogc.pth")

    model = load_model(CONFIG_PATH, WEIGHTS_PATH)

    for caption_column in caption_columns:
        output_column = f"{caption_column}_dino"
        dataframe[output_column] = dataframe.progress_apply(lambda x: dino_predict(x, model, image_directory, caption_column, device), axis=1)

    return dataframe