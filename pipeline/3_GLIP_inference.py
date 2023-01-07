import argparse
import pickle
from tqdm import tqdm
from PIL import Image
import numpy as np
from os.path import join
import os 

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo


def inference(row, glip_demo, args):
    if row['filename'] is None:
        return []

    filename = join(args.image_directory, row['filename'])

    if not os.path.exists(filename):
        return []

    PIL_image = Image.open(filename).convert('RGB')    
    image = np.array(PIL_image)[:, :, [2, 1, 0]]
    expression = row[args.expression_column]
        
    if len(expression) > 60:
        expression = expression[:60]
    if len(expression) == 0:
        return []

    _, results = glip_demo.run_on_web_image(image, expression)

    return postprocess([results, glip_demo.entities])

def postprocess(glip_array):
    """
    Transforms a GLIP array into a MDETR-format array
    :param glip_array: array of inference of GLIP
    :return: the array in a MDETR-format
    """

    caption = [glip_array[1][k - 1] if k < len(glip_array[1]) else glip_array[1][len(glip_array[1]) - 1] for k in
                glip_array[0].get_field('labels')]
    return [glip_array[0].get_field('scores'), glip_array[0].bbox, caption]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to the input file")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument("--expression_column", type=str, help="Column to translate")
    parser.add_argument("--image_directory", type=str, help="Path to the image directory")
    parser.add_argument("--inference_column", type=str, help="Column to store the inference")
    parser.add_argument("--config_file", type=str, help="Path to the config file")
    parser.add_argument("--weights_file", type=str, help="Path to the weights file")
    args = parser.parse_args()

    # update the config options with the config file
    # manual override some options
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", args.weights_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )

    tqdm.pandas()

    # Load the data
    data = pickle.load(open(args.input_file, 'rb'))
    
    # Inference
    data[args.inference_column] = data.progress_apply(lambda row: inference(row, glip_demo, args), axis=1)

    # Save the data
    with open(args.output_file, "wb") as f:
        pickle.dump(data, f)


    
