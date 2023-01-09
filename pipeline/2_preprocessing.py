# Description: Preprocess the data and link the images to the data
# Author: Théo Gieruc
# Date: 2022-01-06

import argparse
import pickle
from os import listdir
from os.path import isfile, join


def preprocess_text(text):
    if text is None:
        return ''
    text = text.lower()
    text = text.replace('portrait of ', '')
    text = text.replace('photograph of ', '')
    text = text.replace('black and white photo of ', '')
    text = text.replace('a group of','')
    text = text.replace('group of','')
    text = text.replace('canton of fribourg','')
    text = text.replace('[fribourg]','')
    text = text.replace('of fribourg','')
    text = text.replace('fribourg','')
    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to the input file")
    parser.add_argument("--id_column", type=str, help="Column to translate")
    parser.add_argument("--image_directory", type=str, help="Path to the image directory")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument("--quality", type=str, help="Quality of the images [324w | 2975h]")
    parser.add_argument("--columns_to_preprocess",  nargs='+', help="Columns to preprocess")

    args = parser.parse_args()

    # Load the data
    data = pickle.load(open(args.input_file, 'rb'))

    # Get the list of images
    images = [f for f in listdir(args.image_directory) if isfile(join(args.image_directory, f))]

    # Filter according to quality
    images = list(filter(lambda k: args.quality in k, images))

    # Link the id to the files
    data['filename'] = data[args.id_column].apply(lambda x:  (list(filter(lambda k: x.lower() in k.lower(), images)))[0].replace(args.image_directory + '/','') if len(list(filter(lambda k: x.lower() in k.lower(), images))) > 0 else None)

    # preprocessing of titles and captions
    for column in args.columns_to_preprocess:
        data[f'{column}_preprocessed'] = data[column].apply(preprocess_text)

    # Save the data
    with open(args.output_file, "wb") as f:
        pickle.dump(data, f)

