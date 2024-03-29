import pickle 
from tqdm import tqdm
import numpy as np
from PIL import Image
import os, sys
import torch
import gdown
from copy import deepcopy
# add file to path
sys.path.append(os.path.dirname((os.path.abspath(__file__))))
from ASM import SegmentationModel
from segment_anything import sam_model_registry, SamPredictor


def ASM_apply_segmentation(row, segmentation_model, args, col):
    data = deepcopy(row[col])

    try:

        filename = os.path.join(args.image_dir, row['filename'])

        image = np.array(Image.open(filename).convert('RGB'))

        segmentation_masks = segmentation_model.single_inference(image, data)

        data['segmentation'] = segmentation_masks
    
    except Exception as e:
        print(e)

    return data


def run_ASM(dataframe, args):
    tqdm.pandas()

    if isinstance(args.detection_columns, str):
        args.detection_columns = [args.detection_columns]

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'ASM_model.pth')
    if not os.path.exists(model_path):
        print('Downloading ASM model weights...')
        gdown.download('https://drive.google.com/uc?id=1OWH7arM-qllbCJwqkMVy9NKWHB398iol', model_path, quiet=False)

    model = SegmentationModel(model_path, device=args.device)

    output_columns = []
    for col in args.detection_columns:
        dataframe[f'{col}_ASM'] = dataframe.progress_apply(lambda x: ASM_apply_segmentation(x, model, args, col), axis=1)
        output_columns.append(f'{col}_ASM')

    return dataframe, output_columns


def SAM_apply_segmentation(row, predictor, args, col):

    data = deepcopy(row[col])

    try:

        image_filename = os.path.join(args.image_dir, row['filename'])

        image = np.array(Image.open(image_filename).convert('RGB'))
        predictor.set_image(image)

        bounding_boxes = data["bounding_boxes"]

        masks = []
        for box in bounding_boxes:
            input_box = box.int().numpy().flatten()
            mask, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False,
            )
            masks.append(torch.from_numpy(mask))
        data['segmentation'] = torch.stack(masks).squeeze(1).squeeze(1)
    
    except Exception as e:
        print(e)
        
    return data

def run_SAM(dataframe, args, algorithm):
    weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    if algorithm == 'SAM-B':
        model_type = 'vit_b'
        weights_path = os.path.join(weights_dir, 'sam_vit_b_01ec64.pth')
        if not os.path.exists(weights_path):
            print('Downloading SAM-B model weights...')
            os.system(f'wget -P {weights_dir} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O {weights_path}')
    elif algorithm == 'SAM-L':
        model_type = 'vit_l'
        weights_path = os.path.join(weights_dir, 'sam_vit_l_0b3195.pth')
        if not os.path.exists(weights_path):
            print('Downloading SAM-L model weights...')
            os.system(f'wget -P {weights_dir} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O {weights_path}')
    elif algorithm == 'SAM-H':
        model_type = 'vit_h'
        weights_path = os.path.join(weights_dir, 'sam_vit_h_4b8939.pth')
        if not os.path.exists(weights_path):
            print('Downloading SAM-H model weights...')
            os.system(f'wget -P {weights_dir} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O {weights_path}')
    else: 
        raise ValueError('Algorithm must be one of SAM-B, SAM-L, SAM-H')
    
    sam = sam_model_registry[model_type](checkpoint=weights_path)
    sam.to(device=args.device)

    predictor = SamPredictor(sam)

    tqdm.pandas()

    columns = args.detection_columns
    if isinstance(columns, str):
        columns = [columns]

    output_columns = []
    for cols in columns:
        dataframe[f'{cols}_{algorithm}'] = dataframe.progress_apply(lambda x: SAM_apply_segmentation(x, predictor, args, cols), axis=1)
        output_columns.append(f'{cols}_{algorithm}')

    return dataframe, output_columns