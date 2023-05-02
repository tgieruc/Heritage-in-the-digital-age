import os, sys
from tqdm import tqdm
import torch
from torchvision.ops import box_convert
tqdm.pandas()

from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict

home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def dino_predict(row, model, image_dir, caption_col, device, box_thresh, text_thresh):

    try:

        filename = os.path.join(image_dir, row['filename'])

        text_prompt = row[caption_col]

        image_source, image = load_image(filename)

        boxes, logits, labels = predict(
            model=model, 
            image=image, 
            caption=text_prompt, 
            box_threshold=box_thresh, 
            text_threshold=text_thresh
        )

        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").int()

        return {'bounding_boxes': xyxy, 'scores': logits, 'labels': labels}

    except Exception as e:
        print(e)
        return {'bounding_boxes': torch.tensor([]), 'scores': torch.tensor([]), 'labels': []}

def run_DINO(algorithm, dataframe, image_directory, caption_columns, device, box_thresh, text_thresh, progress):

    model = algorithm.split("-")[1]
    if model == "SwinT":
        CONFIG_PATH =  os.path.join(home_dir,"submodules/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        WEIGHTS_PATH = os.path.join(home_dir,"submodules/GroundingDINO/model/groundingdino_swint_ogc.pth")
        if not os.path.isfile(WEIGHTS_PATH):
            if not os.path.isdir(os.path.join(home_dir,"submodules/GroundingDINO/model")):
                os.mkdir(os.path.join(home_dir,"submodules/GroundingDINO/model"))
            print('Downloading GroundingDINO SwinT model weights...')
            os.system(f'wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O {WEIGHTS_PATH}')

    elif model == "SwinB":
        CONFIG_PATH =  os.path.join(home_dir,"submodules/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py")
        WEIGHTS_PATH = os.path.join(home_dir,"submodules/GroundingDINO/model/groundingdino_swinb_cogcoor.pth")
        if not os.path.isfile(WEIGHTS_PATH):
            print('Downloading GroundingDINO SwinB model weights...')
            os.system(f'wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth -O {WEIGHTS_PATH}')


    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    output_columns = []
    for caption_column in caption_columns:
        output_column = f"{caption_column}-{algorithm}"
        dataframe[output_column] = dataframe.progress_apply(lambda x: dino_predict(x, model, image_directory, caption_column, device, box_thresh, text_thresh), axis=1)
        output_columns.append(output_column)

    return dataframe, output_columns