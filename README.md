<h1 align="center">Heritage in the digital age</h1>
<h3 align="center"><em>A Step-by-Step Guide to Augmenting Digitized Historical Images </em></h3>
<h4 align="center"><i>Semester Project @ CVlab &amp; EPFL+ECAL Lab </i></h4>
<h4 align="center"><a href="https://tgieruc.github.io/Heritage-in-the-digital-age/">Visit the Website for the gallery and an explanation of the project!</a></h4>

<br>

 A demo of the whole pipeline is available on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tgieruc/Heritage-in-the-digital-age/blob/main/demo_pipeline.ipynb)

The report can be found [here](Report.pdf).


---
## Introduction

This project aims to valorize digitized heritage collections by augmenting images with state-of-the-art computer vision and natural language processing techniques. It is a collaboration between the EPFL+ECAL Lab and the Cantonal University Library of the Canton of Fribourg (BCUFR). The dataset used consists of 2,216 pictures from 1870 to 2003 with AI-generated captions and titles in French.

## The pipeline

The pipeline consists of six steps and takes an image with its French title and AI-generated English caption as input. It produces object-segmented images as output.

1. Translate French titles to English using a pretrained machine translation model (MarianMT).
2. Preprocess captions and titles for phrase grounding.
3. Run phrase grounding on the dataset using two state-of-the-art models (GLIP and MDETR).
4. Postprocess phrase grounding results with non-maximum suppression.
5. Select the best phrase grounding results manually using a GUI or automatically.
6. Segment detected objects in the images. A demo of the segmentation process can be found here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tgieruc/Heritage-in-the-digital-age/blob/main/pipeline/notebooks/segmentation_demo.ipynb)!



## Usage

To run the pipeline, follow these steps:

1. Clone the repository with ```git clone --recurse-submodules https://github.com/tgieruc/Heritage-in-the-digital-age``` and navigate to the directory.
2. Download the dataset from the BCUFR website and place it in the `data` directory. Download the weight file for the segmentation model and place it in the `model` directory.
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
python pipeline/4_apply_nms.py --input_file data/3_phrase_grounding.pkl --output_file data/4_postprocess.pkl --columns_to_process GLIP_caption GLIP_title MDETR_caption MDETR_title
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


## In-depth documentation


For this example, you need the following architecture:

```
.
├── data
│   ├── BCU_database
│   │   ├── original_data.pkl
│   │   └── images
│   │       ├── ALBL00222_2k_324w.jpg
│   │       ├── ALBL00222_2k_2975h.jpg
│   │       └── [...].
├── pipeline
│   ├── src
│   │   └── GLIP [submodule]
│   ├── 1_translate.py
│   ├── 2_preprocessing.py
│   ├── 3_GLIP_inference.py
│   ├── 3_MDETR_inference.py
│   ├── 4_apply_nms.py
│   ├── 5_automatic_selection.py
│   ├── 5_manual_selection.py
│   └── 6_segmentation.py
```

`original_data.pkl` is a Pandas DataFrame serialized as a Pickle file that contains the following columns: *bcu_id* the ID of the image, *title* the title in French and *caption* the AI-generated alternative caption.

The images are stored in the `data/BCU_database/images/` folder and are in two resolutions: 324w and 2975h.

### 1. Translation
The following command is a Python script that translates the values in a specified column of a dataset stored in a pickle file:

<pre><code>python pipeline/1_translate.py --input_file data/BCU_database/original_data.pkl --column title --output_file data/1_translation.pkl --device cuda
</code></pre>

This script takes several arguments:

* `--input_file` specifies the input file, which is a pickle file containing the dataset to be translated.
* `--column` specifies the column in the dataset that should be translated. In this case, the title column will be translated.
* `--output_file` specifies the output file, which is a pickle file where the translated dataset will be stored.
* `--device` specifies the device to use for translation. The value of this argument, cuda, indicates that a GPU should be used if one is available. If you do not want to use the GPU, you should write cpu.

This script adds a column `title_en` to the DataFrame, containing all the titles translated in English.

### 2. Preprocessing
<p>The following command is a Python script that performs the preprocessing of the titles and captions and links each <code>bcu_id</code> to an image:</p>
<pre><code>python pipeline/2_preprocessing.py --input_file data/1_translation.pkl --id_column bcu_id --image_directory data/BCU_database/images --output_file data/2_preprocessing.pkl --quality 324w --columns_to_preprocess caption title_en
</code></pre>
<p>This script takes several arguments:</p>
<ul>
<li><code>--input_file</code> specifies the input file, which is a pickle file containing the dataset to be preprocessed.</li>
<li><code>--id_column</code> specifies the column in the dataset that contains the unique identifier for each record.</li>
<li><code>--image_directory</code> specifies the directory containing the images associated with the ids in the dataset.</li>
<li><code>--output_file</code> specifies the output file, which is a pickle file where the preprocessed dataset will be stored.</li>
<li><code>--quality</code> specifies the quality of the images. It can either be <code>324w</code> or <code>2975h</code>.</li>
<li><code>--columns_to_preprocess</code> specifies a list of columns in the dataset that should be preprocessed. In this case, the <code>caption</code> and <code>title_en</code> columns will be preprocessed.</li>
</ul>
<p>This script creates three new columns <code>caption_preprocessed</code> and <code>title_en_preprocessed</code> containing the preprocessed expressions, and <code>filename</code> containing the filename of the images linked to the <code>bcu_id</code>.</p>

### 3. Phrase Grounding
<p>As this part is the most computationally intensive and requires the most GPU memory, Two Jupyter notebooks are available for execution on Google Colab, but it can also be run locally.</p>
<p>The first two commands run the <code>pipeline/3_MDETR_inference.py</code> script to perform inference using MDETR on the <code>caption_preprocessed</code> and <code>title_en_preprocessed</code> columns of the dataset:</p>
<pre><code>python pipeline/3_MDETR_inference.py --input_file data/2_preprocessing.pkl --output_file temp.pkl --expression_column caption_preprocessed --inference_column MDETR_caption --image_directory data/BCU_database/images
</code></pre>
<pre><code>python pipeline/3_MDETR_inference.py --input_file temp.pkl --output_file temp.pkl --expression_column title_en_preprocessed --inference_column MDETR_title --image_directory data/BCU_database/images
</code></pre>
<p>The third command uses wget to download a pretrained model from a specified URL and save it to the <code>model</code> directory as <code>glip_large_model.pth</code>.</p>
<pre><code>wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O model/glip_large_model.pth
</code></pre>


The final two commands run the `python pipeline/3_GLIP_inference.py` script to perform inference on the `caption_preprocessed` and `title_en_preprocessed` columns of the dataset using the pretrained GLIP model:

<pre><code>python pipeline/3_GLIP_inference.py --input_file temp.pkl --output_file temp.pkl --expression_column caption_preprocessed --inference_column GLIP_caption --image_directory data/BCU_database/images --config_file pipeline/src/GLIP/configs/pretrain/glip_Swin_L.yaml --weights_file model/glip_large_model.pth
</code></pre>

<pre><code>python pipeline/3_GLIP_inference.py --input_file temp.pkl --output_file data/3_phrase_grounding.pkl --expression_column title_en_preprocessed --inference_column GLIP_title --image_directory data/BCU_database/images --config_file pipeline/src/GLIP/configs/pretrain/glip_Swin_L.yaml --weights_file model/glip_large_model.pth
</code></pre>

<p>These scripts take several arguments:</p>
<ul>
  <li><code>--input_file</code> specifies the input file, which is a pickle file containing the dataset on which to perform inference.</li>
  <li><code>--output_file</code> specifies the output file, which is a pickle file where the resulting dataset will be stored.</li>
  <li><code>--expression_column</code> specifies the column in the dataset containing the expressions to be grounded.</li>
  <li><code>--inference_column</code> specifies the column in the resulting dataset where the grounded expressions will be stored.</li>
  <li><code>--image_directory</code> specifies the directory containing the images associated with the records in the dataset.</li>
  <li><code>--config_file</code> (for the GLIP model only) specifies the configuration file to use for the model.</li>
  <li><code>--weights_file</code> (for the GLIP model only) specifies the file containing the pretrained weights for the model.</li>
</ul>

This will add four columns, for each combination of GLIP MDETR with caption or title, with the results of the inferences.

### 4. Postprocessing
The following command is a Python script that applies non-maximum suppression (NMS) to the values in specified columns of a dataset stored in a pickle file:

<pre><code>python pipeline/4_apply_nms.py --input_file data/3_phrase_grounding.pkl --output_file data/4_postprocess.pkl --columns_to_process GLIP_caption GLIP_title MDETR_caption MDETR_title
</code></pre>
This script takes several arguments:

<ul>
  <li><code>--input_file</code> specifies the input file, which is a pickle file containing the dataset on which to apply NMS.</li>
  <li><code>--output_file</code> specifies the output file, which is a pickle file where the resulting dataset will be stored.</li>
  <li><code>--columns_to_process</code> specifies a list of columns in the dataset on which to apply NMS. In this case, NMS will be applied to the <code>GLIP_caption</code>, <code>GLIP_title</code>, <code>MDETR_caption</code>, and <code>MDETR_title</code> columns.</li>
</ul>
This applies the NMS directly to the four columns containing the phrase grounding results.


### 5. Selecting the best phrase grounding
The following commands are Python scripts that select the best phrase grounding for each expression in a dataset stored in a pickle file. One script performs automatic selection and the other provides a graphical user interface (GUI) for manual selection.

The automatic selection algorithm uses a expression ranking model. It chooses GLIP if available, and selects the highest-ranked expression.

This command runs the <code>pipeline/5_automatic_selection_best_phrase_grounding.py</code> script to perform automatic selection:

<pre><code>python pipeline/5_automatic_selection_best_phrase_grounding.py \
--input_file data/4_postprocess.pkl \
--output_file data/5_automatic.pkl \
--selection_column automatic_selection
</code></pre>

This command runs the <code>pipeline/5_manual_selection_best_phrase_grounding.py</code> script to provide a GUI for manual selection:

<pre><code>python pipeline/5_manual_selection_best_phrase_grounding.py \
--input_file data/4_postprocess.pkl \
--selection_column manually_selected \
--output_file data/5_manual.pkl \
--image_directory data/BCU_database/images
</code></pre>

These scripts take several arguments:

<ul>
  <li><code>--input_file</code> specifies the input file, which is a pickle file containing the dataset from which to select the best phrase groundings.</li>
  <li><code>--output_file</code> specifies the output file, which is a pickle file where the resulting dataset will be stored.</li>
  <li><code>--selection_column</code> specifies the column in the resulting dataset where the selected phrase groundings will be stored.</li>
  <li><code>--image_directory</code> (for the manual selection script only) specifies the directory containing the images associated with the records in the dataset.</li>
</ul>

### 6. Segmentation
<p>
The following command is a Python script that performs segmentation on the images associated with a dataset stored in a pickle file:
<pre><code>python pipeline/6_segmentation.py 
  --input_file data/5_automatic.pkl
  --output_dir data/6_segmentation 
  --image_dir data/BCU_database/03_resized 
  --selection_column automatic_selection 
  --save_fig 
  --save_segmentation_pickle 
  --model_path model/model_segmentation.pth 
  --save_colored_text_array 
  --save_colored_text_html 
  --device cuda
</code></pre></p>

<pre>
The following command is a Python script that performs segmentation on the images associated with a dataset stored in a pickle file:
<code>
python pipeline/6_segmentation.py --input_file data/5_automatic.pkl --output_dir data/6_segmentation --image_dir data/BCU_database/03_resized --selection_column automatic_selection --save_fig --save_segmentation_pickle --model_path model/model_segmentation.pth --save_colored_text_array --save_colored_text_html --device cuda
</code>
</pre>
This script takes several arguments:

<ul>
  <li><code>--input_file</code> specifies the input file, which is a pickle file containing the dataset associated with the images on which to perform text segmentation.</li>
  <li><code>--output_dir</code> specifies the directory where the output of the segmentation will be saved.</li>
  <li><code>--image_dir</code> specifies the directory containing the images on which to perform the segmentation.</li>
  <li><code>--selection_column</code> specifies the column in the input dataset containing the selected phrase groundings.</li>
  <li><code>--save_fig</code> specifies that the output images should be saved as figures.</li>
  <li><code>--save_segmentation_pickle</code> specifies that each segmentation will be saved as a pickle file.</li>
  <li><code>--model_path</code> specifies the file containing the pretrained model to use for segmentation.</li>
  <li><code>--save_colored_text_array</code> specifies that the expression colored according to the segmentation should be added to the DataFrame.</li>
  <li><code>--save_colored_text_html</code> specifies that the expression colored using HTML formatting according to the segmentation should be added.</li>
  <li><code>--device</code> specifies the device to use for text segmentation. In this case, the script will use a GPU if available.</li>
</ul>



## Acknowledgements

This project uses the following projects:
* The caption generation pipeline: [Chenkai Wang](https://github.com/cnWangChenkai)
* The English to French translation model: [MarianMT](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en)
* One of the two phrase-grounding model used: [GLIP](https://github.com/microsoft/GLIP)
* The other phrase-grounding model: [MDETR](https://github.com/ashkamath/mdetr)
* The NLP model for ranking the expressions: [DistilBERT](https://arxiv.org/abs/1910.01108)
