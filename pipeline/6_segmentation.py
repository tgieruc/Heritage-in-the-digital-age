import pickle 
import argparse
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from src.segmentation import SegmentationModel, plot_results, colored_text_to_html
from os.path import join
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="Path to the input file")
parser.add_argument("--output_dir", type=str, help="Path to the output directory")
parser.add_argument("--image_dir", type=str, help="Path to the image directory")
parser.add_argument("--selection_column", type=str, help="Column to select")
parser.add_argument("--save_fig",action='store_true', help="Save the figures")
parser.add_argument("--save_segmentation_pickle", action='store_true', help="Save the segmentation")
parser.add_argument("--save_segmentation_pandas", action='store_true', help="Save the segmentation")
parser.add_argument("--model_path", type=str, help="Path to the model", default='models/segmentation_model.pth')
parser.add_argument("--device", type=str, help="Device to use")
parser.add_argument("--save_colored_text_array", action='store_true', help="Save the colored text")
parser.add_argument("--save_colored_text_html", action='store_true', help="Save the colored text")
args = parser.parse_args()

if args.device is None:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

def apply_segmentation(row, segmentation_model):
    filename = join(args.image_dir, row['filename'])
    if not os.path.exists(filename):
        return []
    
    output = {}    
    phrase_grounding = row[row[args.selection_column]]

    data = {'labels': phrase_grounding[2], 'bbox': phrase_grounding[1], 'conf': phrase_grounding[0]}

    image = np.array(Image.open(filename).convert('RGB'))

    segmentation_masks = segmentation_model.single_inference(image, data)

    if (args.save_fig or args.save_colored_text) and (segmentation_masks is not None):
        ax = plt.gca()
        expression = row['caption'] if 'caption' in row[args.selection_column] else row['title_en']
        colored_text = plot_results(image.astype(int), data, expression=expression, segmentations=segmentation_masks, ax=ax, conf=0)
        plt.tight_layout(pad=0)

        if args.save_fig:
            plt.savefig(join(args.output_dir, row['filename']), bbox_inches='tight', pad_inches=0)
        if args.save_colored_text_array:
            output['colored_text'] = colored_text
        if args.save_colored_text_html:
            output['colored_text_html'] = colored_text_to_html(colored_text)
        plt.cla()
    if args.save_segmentation_pickle:
        pickle_name = join(args.output_dir,  row['filename'].replace('.png','').replace('.jpg','') + '.pkl')
        pickle.dump(segmentation_masks, open(pickle_name, 'wb'))
    if args.save_segmentation_pandas:
        output['segmentation'] = segmentation_masks

    return output












if __name__ == '__main__':
    tqdm.pandas()
    
    with open(args.input_file, 'rb') as f:
        data = pickle.load(f)

    # create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data = data.loc[data[args.selection_column].notna()]
    segmentation_model = SegmentationModel(args.model_path, device=args.device)

    data['segmentation_output'] = data.progress_apply(lambda x: apply_segmentation(x, segmentation_model), axis=1)

    keys = []
    if args.save_colored_text_array:
        keys.append('colored_text')
    if args.save_colored_text_html:
        keys.append('colored_text_html')
    if args.save_segmentation_pandas:
        keys.append('segmentation')

    for key in keys:
        data[key] = data['segmentation_output'].apply(lambda x: x[key] if key in x else None)
    
    with open(join(args.output_dir, '7_segmentation_output.pkl'), 'wb') as f:
        pickle.dump(data, f)