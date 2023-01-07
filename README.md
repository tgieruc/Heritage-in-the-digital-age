<h1 align="center">Heritage in the digital age</h1>
<h3 align="center"><i>Semester Project @ CVlab &amp; EPFL+ECAL Lab </i></h3>
<h4 align="center"><a href="https://tgieruc.github.io/Heritage-in-the-digital-age/">Visit our Website for the Gallery!</a></h4>

<br>

---
## Introduction

This project aims to valorize digitized heritage collections by augmenting images with state-of-the-art computer vision and natural language processing techniques. It is a collaboration between the EPFL+ECAL Lab and the Cantonal University Library of the Canton of Fribourg (BCUFR). The dataset used consists of 2,216 pictures from 1870 to 2003 with AI-generated captions and titles in French.

## The pipeline

The pipeline consists of six steps and takes an image with its French title and AI-generated English caption as input. It produces object-segmented images as output.

1. Translate French titles to English using a pretrained machine translation model (MarianMT).
2. Preprocess captions and titles for phrase grounding.
3. Run phrase grounding on the dataset using two state-of-the-art models (GLIP and MDETR).
4. Postprocess phrase grounding results with non-maximum suppression and label correction.
5. Select the best phrase grounding results using a GUI.
6. Segment detected objects in the images.

## Requirements

This project requires Python 3.6 or higher, along with the following libraries:

- PyTorch 1.7.0 or higher
- Hugging Face Transformers 4.6.0 or higher
- OpenCV 4.5.0 or higher

## Usage

To run the pipeline, follow these steps:

1. Clone the repository and navigate to the directory.
2. Download the dataset from the BCUFR website and place it in the `data` directory.
3. Run the following command to translate the titles:
```bash
python pipeline/1_translate.py --input_file data/BCU_database/original_data.pkl --column title --output_file data/1_translation.pkl --device cuda
```
4. Run the following command to preprocess the captions and titles:
```bash
python pipeline/2_preprocessing.py --input_file data/1_translation.pkl --id_column bcu_id --image_directory data/BCU_database/03_resized --output_file data/2_preprocessing.pkl --quality 324w --columns_to_preprocess caption title_en
```
5. Run the following command to perform phrase grounding:
```bash
python pipeline/3_MDETR_inference.py --input_file data/2_preprocessed.pkl --output_file temp.pkl --expression_column caption_preprocessed --inference_column MDETR_caption --image_directory data/BCU_database/03_resized 

python pipeline/3_MDETR_inference.py --input_file temp.pkl --output_file temp.pkl --expression_column title_en_preprocessed --inference_column MDETR_title --image_directory data/BCU_database/03_resized 


wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O model/glip_large_model.pth


python pipeline/3_GLIP_inference.py --input_file temp.pkl --output_file temp.pkl --expression_column caption_preprocessed --inference_column GLIP_caption --image_directory data/BCU_database/03_resized --config_file pipeline/src/GLIP/configs/pretrain/glip_Swin_L.yaml --weights_file model/glip_large_model.pth 

python pipeline/3_GLIP_inference.py --input_file temp.pkl --output_file data/3_phrase_grounding.pkl --expression_column title_en_preprocessed --inference_column GLIP_title --image_directory data/BCU_database/03_resized --config_file pipeline/src/GLIP/configs/pretrain/glip_Swin_L.yaml --weights_file model/glip_large_model.pth 
```

6. Run the following command to postprocess the phrase grounding results:
```bash
python pipeline/4_apply_nms.py --input_file data/3_phrase_grounding.pkl --output_file data/4_postprocess.py --columns_to_process GLIP_caption GLIP_title MDETR_caption MDETR_title
```


7. Run the following command to select the best phrase grounding results
* For automatic selection:
```bash
python pipeline/5_automatic_selection_best_phrase_grounding.py --input_file data/4_postprocess.py --output_file data/5_automatic.pkl --selection_column automatic_selection
```
* For manual selection, using the GUI:
```bash
python pipeline/5_manual_selection_best_phrase_grounding.py --input_file data/4_postprocess.py --output_file data/5_manual.pkl --selection_column manually_selected --image_directory data/BCU_database/03_resized 
```

8. Run the following command to segment the detected objects in the images:
```bash
python pipeline/6_segmentation.py --input_file data/5_automatic.pkl --output_dir data/6_segmentation --image_dir data/BCU_database/03_resized --selection_column automatic_selection --save_fig --save_segmentation_pickle --model_path model/model_segmentation.pth --save_colored_text_array --save_colored_text_html --device cuda
```

