import os
import sys
from types import SimpleNamespace

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import supervision as sv
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import zipfile
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir)
sys.path.append(os.path.join(file_dir, "webui_helpers"))
sys.path.append(os.path.join(file_dir, "submodules"))
sys.path.append(os.path.join(file_dir, "submodules/GroundingDINO"))

from webui_helpers.phrase_grounding import run_DINO
from webui_helpers.segmentation import run_ASM, run_SAM

tqdm.pandas()

dataframe = None
img_path = os.path.join(file_dir, "data", "images")

COLORS = 255 * np.array([[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933], [0,0,0]])


#------------------ Get and update dataframe ------------------#

def get_data_translate(file):
    global dataframe
    dataframe = pd.read_pickle(file.name)
    options = []
    if "title" in dataframe.columns:
        options.append("title")
    return dataframe.head(), gr.Dropdown.update(choices=dataframe.columns.tolist(), value=options)

def update_translate():
    global dataframe
    if dataframe is None:
        dataframe = pd.DataFrame()
    options = []
    if "title" in dataframe.columns:
        options.append("title")
    return dataframe.head(), gr.Dropdown.update(choices=dataframe.columns.tolist(), value=options), img_path

def get_data_preprocess(file):
    global dataframe
    dataframe = pd.read_pickle(file.name)
    options_preprocess = []
    options_id = ""

    if "title_en" in dataframe.columns:
        options_preprocess.append("title_en")
    if "caption" in dataframe.columns:
        options_preprocess.append("caption")
    
    for column in dataframe.columns:
        if column.endswith("_id"):
            options_id = column

    return dataframe.head(), gr.Dropdown.update(choices=dataframe.columns.tolist(), value=options_preprocess), gr.Dropdown.update(choices=dataframe.columns.tolist(), value=options_id)

def update_preprocess():
    global dataframe
    if dataframe is None:
        dataframe = pd.DataFrame()
    options_preprocess = []
    options_id = ""

    if "title_en" in dataframe.columns:
        options_preprocess.append("title_en")
    if "caption" in dataframe.columns:
        options_preprocess.append("caption")
    
    for column in dataframe.columns:
        if column.endswith("_id"):
            options_id = column

    return dataframe.head(), gr.Dropdown.update(choices=dataframe.columns.tolist(), value=options_preprocess), gr.Dropdown.update(choices=dataframe.columns.tolist(), value=options_id), img_path

def get_data_phrase_grounding(file):
    global dataframe
    dataframe = pd.read_pickle(file.name)
    options = []

    for column in dataframe.columns:
        if column.endswith("_preprocessed"):
            options.append(column)

    return dataframe.head(), gr.Dropdown.update(choices=dataframe.columns.tolist(), value=options)

def update_phrase_grounding():
    global dataframe
    global img_path
    if dataframe is None:
        dataframe = pd.DataFrame()
    options = []

    for column in dataframe.columns:
        if column.endswith("_preprocessed"):
            options.append(column)

    return dataframe.head(), gr.Dropdown.update(choices=dataframe.columns.tolist(), value=options), img_path

def get_data_segmentation(file):
    global dataframe
    dataframe = pd.read_pickle(file.name)
    options = []
    for column in dataframe.columns:
        if "_GLIP" in column:
            options.append(column)
        elif "_MDETR" in column:
            options.append(column)
        elif "_DINO" in column:
            options.append(column)

    return dataframe.head(), gr.Dropdown.update(choices=dataframe.columns.tolist(), value=options)

def update_segmentation():
    global dataframe
    global img_path
    if dataframe is None:
        dataframe = pd.DataFrame()
    options = []
    for column in dataframe.columns:
        if "_GLIP" in column:
            options.append(column)
        elif "_MDETR" in column:
            options.append(column)
        elif "_DINO" in column:
            options.append(column)

    return dataframe.head(), gr.Dropdown.update(choices=dataframe.columns.tolist(), value=options), img_path

def get_data_visualization(file):
    global dataframe
    global img_path
    dataframe = pd.read_pickle(file.name)

    options = []

    for column in dataframe.columns:
        if column.endswith("_ASM"):
            options.append(column)
        elif column.endswith("_SAM-H"):
            options.append(column)
        elif column.endswith("_SAM-L"):
            options.append(column)
        elif column.endswith("_SAM-B"):
            options.append(column)

    return dataframe.head(), gr.Dropdown.update(choices=dataframe.columns.tolist(), value=options), img_path

def update_visualization():
    global dataframe
    global img_path
    if dataframe is None:
        dataframe = pd.DataFrame()
    
    options = []

    for column in dataframe.columns:
        if column.endswith("_ASM"):
            options.append(column)
        elif column.endswith("_SAM-H"):
            options.append(column)
        elif column.endswith("_SAM-L"):
            options.append(column)
        elif column.endswith("_SAM-B"):
            options.append(column)

    return dataframe.head(), gr.Dropdown.update(choices=dataframe.columns.tolist(), value=options), img_path

def save_dataframe(directory):
    global dataframe
    if dataframe is None:
        return "No dataframe loaded"

    dataframe.to_pickle(directory)

    return "Dataframe saved!"

def upload_images(file):
    global img_path
    img_path = os.path.join(file_dir, "data", "images")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    
    with zipfile.ZipFile(file.name, 'r') as zip_ref:
        zip_ref.extractall(img_path)

    return gr.UploadButton().update("Images uploaded!"), img_path

#----------------- Module functions -----------------#

def translate_titles(columns, language):
    def translate(sentence, model, tokenizer, device):
        input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    global dataframe
    if dataframe is None:
        return "No dataframe loaded"
    
    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = {
        "French": "Helsinki-NLP/opus-mt-fr-en",
        "German": "Helsinki-NLP/opus-mt-de-en",
        "Italian": "Helsinki-NLP/opus-mt-it-en",
        "Spanish": "Helsinki-NLP/opus-mt-es-en",
        "Dutch": "Helsinki-NLP/opus-mt-nl-en",
        "Portuguese": "Helsinki-NLP/opus-mt-pt-en",
        "Russian": "Helsinki-NLP/opus-mt-ru-en"
    }
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name[language]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name[language])


    if isinstance(columns, str):
        columns = [columns]
    
    for column in columns:
        # check that the column to translate exists
        assert column in dataframe.columns, f"Can't find the column {column}"
        
        # translate the data
        dataframe[f'{column}_en'] = dataframe[column].progress_apply(lambda x: translate(x, model, tokenizer, device) if pd.notna(x) else '')

    return dataframe

def preprocess(columns="title_en"):
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
    global dataframe
    if dataframe is None:
        return "No dataframe loaded"
    
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        # check that the column to translate exists
        assert column in dataframe.columns, f"Can't find the column {column}"

        # preprocess the data
        dataframe[f'{column}_preprocessed'] = dataframe[column].progress_apply(lambda x: preprocess_text(x) if pd.notna(x) else '')

    return dataframe

def get_image_names(directory, id_column, quality):
    global dataframe

    images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Filter according to quality
    if quality != "All":
        images = list(filter(lambda k: quality in k, images))

    if isinstance(id_column, list):
        id_column = id_column[0]

    # Link the id to the files
    dataframe['filename'] = dataframe[id_column].apply(lambda x:  (list(filter(lambda k: x.lower() in k.lower(), images)))[0].replace(directory + '/','') if len(list(filter(lambda k: x.lower() in k.lower(), images))) > 0 else None)

    return dataframe

def run_phrase_grounding(algorithm, image_directory, caption_columns, device):
    global dataframe

    if dataframe is None:
        return "No dataframe loaded"
    
    if algorithm == "MDETR":
        # dataframe = run_MDETR(dataframe, image_directory, caption_column, device)
        pass
    elif algorithm == "Grounding DINO":
        dataframe = run_DINO(dataframe, image_directory, caption_columns, device)


    return dataframe

def run_segmentation(algorithm, image_dir, detection_columns, device):
    global dataframe

    if dataframe is None:
        return "No dataframe loaded"
    
    args = SimpleNamespace()
    args.image_dir = image_dir
    args.detection_columns = detection_columns
    args.device = device
    if algorithm == "ASM":
        dataframe = run_ASM(dataframe, args)
    elif algorithm.split("-")[0] == "SAM":
        dataframe = run_SAM(dataframe, args, algorithm)

    return dataframe

def link_labels_to_colors(labels):
    label_set = set(labels)
    colors = {}
    for i, label in enumerate(label_set):
        colors[label] = COLORS[i]
    
    return colors

def apply_mask(image, mask, color, alpha=0.3):
    """Apply the given mask to the image.

    Args:
        image (numpy.ndarray): The image to apply the mask to.
        mask (numpy.ndarray): The mask to apply to the image.
        color (list or tuple): The color to use for the mask.
        alpha (float, optional): The transparency of the mask. Default is 0.3.

    Returns:
        numpy.ndarray: The image with the applied mask.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def visualize_img(img, element, visu_selection, data_column, fontscale):
    data = element[data_column]

    masks = data["segmentation"].numpy() if "segmentation" in data.keys() else None
    bounding_boxes = data["bounding_boxes"].numpy().reshape(-1,4) if "bounding_boxes" in data.keys() else None
    scores = data["scores"].numpy().flatten() if "scores" in data.keys() else None
    labels = data["labels"] if "labels" in data.keys() else None


    if ("Score" in visu_selection) and ("Label" in visu_selection):
        labels = [
            f"{phrase} {logit:.2f}"
            for phrase, logit
            in zip(labels, scores)
        ]
    elif "Score" in visu_selection:
        labels = [
            f"{logit:.2f}"
            for logit
            in scores
        ]
    elif "Label" in visu_selection:
        labels = labels

    if ("Bounding box" in visu_selection) and (bounding_boxes is not None):
        if scores is None:
            detections = sv.Detections(xyxy=bounding_boxes)
        else:
            detections = sv.Detections(xyxy=bounding_boxes, confidence=scores)

        box_annotator = sv.BoxAnnotator(text_scale=fontscale, text_padding=0)
        if labels is None:
            labels = ["" for _ in range(len(bounding_boxes))]
        img = box_annotator.annotate(scene=img, detections=detections, labels=labels)
    
    if ("Segmentation" in visu_selection) and (masks is not None):
        detections = sv.Detections(xyxy=bounding_boxes, mask=masks)
        mask_annotator = sv.MaskAnnotator()
        img = mask_annotator.annotate(scene=img, detections=detections)

    return img

def visualize_dataframe(img_dir, num_imgs, data_columns, visu_selection, fontscale):
    global dataframe
    
    if num_imgs > len(dataframe):
        num_imgs = len(dataframe)

    # will be using temp to s   tore the images
    temp_folder = os.path.join(file_dir, "temp")
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    # get highest number.png in temp folder
    files = [file.split(".")[0] for file in os.listdir(temp_folder) if file.endswith(".png")]
    if len(files) == 0:
        id_offset = 0
    else:
        id_offset = max([int(file) for file in files]) + 1

    for i, element in dataframe[:num_imgs].iterrows():

        img = cv2.imread(os.path.join(img_dir, element['filename']))

        if data_columns == '' or data_columns is None:
            cv2.imwrite(os.path.join(temp_folder, f"{i + id_offset}.png"), img)
            continue
        
        if isinstance(data_columns, str):
            data_columns = [data_columns]

        imgs = []

        for data_column in data_columns:
            imgs.append(visualize_img(img.copy(), element, visu_selection, data_column, fontscale))
        

        if len(imgs) == 0:
            imgs = [img]
        img = np.concatenate(imgs, axis=1)

        


        cv2.imwrite(os.path.join(temp_folder, f"{i + id_offset}.png"), img)

    html = f"<div id='{np.random.rand()}'>"
    for i in range(num_imgs):
        html += f"<img id='{np.random.rand()}' src='file/{file_dir}/temp/{i + id_offset}.png' height='400'/>"
    html += "</div>"
    return gr.HTML.update(html)

# ------------------------- GRADIO ------------------------- #


with gr.Blocks() as demo:

    #------------------------ TRANSLATE ------------------------#

    with gr.Tab("Translate") as tab_translate:

        with gr.Column():

            df = gr.DataFrame()


            with gr.Row():

                with gr.Column():
                    upload_button = gr.UploadButton("Click to Upload the dataframe", type="file")
                    upload_images_button = gr.UploadButton("Upload images in zip", type="file", file_types=["zip"])
                    nothing = gr.Button("Nothing", visible=False)
                    upload_images_button.upload(upload_images, upload_images_button, [upload_images_button, nothing])

                with gr.Column():
                    language = gr.Dropdown(["French", "German", "Italian", "Spanish", "Dutch", "Portuguese", "Russian"], label="Language", value="French")
                    column_to_translate = gr.Dropdown(label="Column(s) to translate", multiselect=True)
                    upload_button.upload(get_data_translate, upload_button, [df, column_to_translate])

                    # column_to_translate = gr.Text("title", label="Column to translate")
                    translate_button = gr.Button("Translate")
                    translate_button.click(translate_titles, [column_to_translate, language], df)

                with gr.Column():
                    save_directory = gr.Text("df_translated.pkl", label="Save directory")
                    save_button = gr.Button("Save")
                    save_button.click(save_dataframe, save_directory, save_button)

            tab_translate.select(update_translate, [], [df, column_to_translate, nothing])

    # ------------------------- PREPROCESS ------------------------- #

    with gr.Tab("Preprocessing") as tab_preprocessing:

        with gr.Column():

            df = gr.DataFrame()

            with gr.Row():
                    
                    with gr.Column():
                        upload_button = gr.UploadButton("Click to Upload the dataframe", type="file")
                        upload_images_button = gr.UploadButton("Upload images in zip", type="file")

                    with gr.Column():
                        column_to_preprocess = gr.Dropdown(label="Column(s) to preprocess", multiselect=True)
                        translate_button = gr.Button("Preprocess")
                        translate_button.click(preprocess, column_to_preprocess, df)

                    with gr.Column():
                        image_dir = gr.Text("demo/img/", label="Image directory")
                        id_column = gr.Dropdown(label="ID column", multiselect=False)
                        quality = gr.Radio(["All", "324w", "2975h"], label="Quality", value="All")
                        get_image_names_button = gr.Button("Get image names")
                        get_image_names_button.click(get_image_names, [image_dir, id_column, quality], df)
                        upload_images_button.upload(upload_images, upload_images_button, [upload_images_button, image_dir])

                    with gr.Column():
                        save_directory = gr.Text("df_preprocessed.pkl", label="Save directory")
                        save_button = gr.Button("Save")
                        save_button.click(save_dataframe, save_directory, save_button)
        
                    upload_button.upload(get_data_preprocess, upload_button, [df, column_to_preprocess, id_column])

            tab_preprocessing.select(update_preprocess, [], [df, column_to_preprocess, id_column, image_dir])

    # ------------------------- PHRASE GROUNDING ------------------------- #

    with gr.Tab("Phrase Grounding") as tab_phrase_grounding:
        
        with gr.Column():

            df = gr.DataFrame()

            with gr.Row():

                with gr.Column():
                    upload_button = gr.UploadButton("Click to Upload the dataframe", type="file")
                    upload_images_button = gr.UploadButton("Upload images in zip", type="file", file_types=["zip"])

                with gr.Column():
                    algorithm = gr.Dropdown(["Grounding DINO"], label="Algorithm", multiselect=False, value="Grounding DINO")
                    image_dir = gr.Text("demo/img/", label="Image directory")
                    caption_column = gr.Dropdown(label="Column for caption", multiselect=True)
                    devices = ["cpu"]
                    if torch.cuda.is_available():
                        devices.append("cuda")
                    device = gr.Radio(devices, label="Device", value="cuda" if torch.cuda.is_available() else "cpu")
                    run_button = gr.Button("Run")
                    run_button.click(run_phrase_grounding, [algorithm, image_dir, caption_column, device], df)
                    upload_images_button.upload(upload_images, upload_images_button, [upload_images_button, image_dir])

                upload_button.upload(get_data_phrase_grounding, upload_button, [df, caption_column])

                with gr.Column():
                    save_directory = gr.Text("df_phrase_grounding.pkl", label="Save directory")
                    save_button = gr.Button("Save")
                    save_button.click(save_dataframe, save_directory, save_button)
            
            tab_phrase_grounding.select(update_phrase_grounding, [], [df, caption_column, image_dir])

    # ------------------------- OBJECT SEGMENTATION ------------------------- #

    with gr.Tab("Segmentation") as tab_segmentation:

        with gr.Column():

            df = gr.DataFrame()

            with gr.Row():

                with gr.Column():
                    upload_button = gr.UploadButton("Click to Upload the dataframe", type="file")
                    upload_images_button = gr.UploadButton("Upload images in zip", type="file", file_types=["zip"])

                with gr.Column():
                    # save_options = gr.CheckboxGroup(["PNG", "PICKLE", "PANDAS"], label="Save options", value=["PNG", "PICKLE", "PANDAS"])
                    algorithm = gr.Radio(["ASM", "SAM-B", "SAM-L", "SAM-H"], label="Algorithm", value="SAM-B")
                    image_dir = gr.Text("demo/img/", label="Image directory")
                    # output_dir = gr.Text("demo/img_result/", label="Output directory")
                    detection_columns = gr.Dropdown(label="Column for segmentation", multiselect=True)
                    model_dir = os.path.join(os.getcwd(), "model")
                    if not os.path.exists(model_dir):
                        os.mkdir(model_dir)
                    devices = ["cpu"]
                    if torch.cuda.is_available():
                        devices.append("cuda")
                    device = gr.Radio(devices, label="Device", value="cuda" if torch.cuda.is_available() else "cpu")
                    run_button = gr.Button("Run")
                    run_button.click(run_segmentation, [algorithm, image_dir, detection_columns, device], df)
                
                upload_button.upload(get_data_segmentation, upload_button, [df, detection_columns])
                upload_images_button.upload(upload_images, upload_images_button, [upload_images_button, image_dir])

                with gr.Column():
                    save_directory = gr.Text("df_segmented.pkl", label="Save directory")
                    save_button = gr.Button("Save")
                    save_button.click(save_dataframe, save_directory, save_button)

            tab_segmentation.select(update_segmentation, [], [df, detection_columns, image_dir])

    # ------------------------- VISUALIZATION ------------------------- #

    with gr.Tab("Visualization") as tab_visualization:

        with gr.Column():

            df = gr.DataFrame()

            with gr.Row():
                with gr.Column():
                    upload_button = gr.UploadButton("Click to Upload the dataframe", type="file")
                    upload_images_button = gr.UploadButton("Upload images in zip", type="file", file_types=["zip"])

                with gr.Column():
                    image_dir = gr.Text("demo/img/", label="Image directory")
                    data_columns = gr.Dropdown(label="Column for data", multiselect=True)
                    visu_selection = gr.CheckboxGroup(["Label", "Score", "Bounding box", "Segmentation"], label="Data to visualize", value=["Label", "Score", "Bounding box", "Segmentation"])
                    num_samples = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Number of samples")
                    font_scale = gr.Slider(minimum=0.1, maximum=2, value=0.5, step=0.1, label="Font scale")
                    run_button = gr.Button("Run")
            
            visuHTML = gr.HTML()

            upload_button.upload(get_data_visualization, upload_button, [df, data_columns])
            upload_images_button.upload(upload_images, upload_images_button, [upload_images_button, image_dir])

            run_button.click(visualize_dataframe, [image_dir, num_samples, data_columns, visu_selection, font_scale], visuHTML)

            tab_visualization.select(update_visualization, [], [df, data_columns, image_dir])



demo.launch(share=True)


