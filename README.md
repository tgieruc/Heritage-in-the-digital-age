<h1 align="center">Heritage in the digital age</h1>
<h3 align="center"><em>A Step-by-Step Guide to Augmenting Digitized Historical Images, <br> with the help of <a href="https://github.com/salesforce/LAVIS">LAVIS</a>, <a href="https://github.com/IDEA-Research/GroundingDINO">groundingDINO</a> and <a href="https://github.com/facebookresearch/segment-anything">Segment Anything</a> </em></h3>

<h4 align="center"><i>Semester Project @ CVlab &amp; EPFL+ECAL Lab </i></h4>
<h4 align="center"><a href="https://tgieruc.github.io/Heritage-in-the-digital-age/">Visit the Website for the gallery and an explanation of the project!</a></h4>
<p align="center"><a  href="https://colab.research.google.com/github/tgieruc/Heritage-in-the-digital-age/blob/main/WebUI_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" width=150px alt="Open In Colab"/></a></p>
<br>

The project in its original state can be found [here](https://github.com/tgieruc/Heritage-in-the-digital-age/releases/tag/v1.0).


---

WebUI designed for **augmenting** digitized historical images by **generating captions**,  **grounding** the captions and **segmenting** their content. 


## :bulb: Highlights
* Simple WebUI using <a href=https://gradio.app/><img src="https://warehouse-camo.ingress.cmh1.psfhosted.org/a95ef5913dc4cc84d2155ff690a0fa0d4c33d7e2/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f67726164696f2d6170702f67726164696f2f6d61696e2f726561646d655f66696c65732f67726164696f2e737667" height=20px alt="Gradio"/></a>
* Caption generation using BLIP and BLIP2 <a href=https://github.com/salesforce/LAVIS ><img src="https://github.com/salesforce/LAVIS/raw/main/docs/_static/logo_final.png" height=20px alt="LAVIS"/></a>
* Translation of captions to English using <a src=https://huggingface.co/Helsinki-NLP>Helsinki-NLP<img src=https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png height=20px alt="HuggingFace"/></a>
* Grounding of captions using [:sauropod: groundingDINO](https://github.com/IDEA-Research/GroundingDINO)
* Segmentation of images using <a href=https://github.com/facebookresearch/segment-anything>Segment Anything <img src=https://seeklogo.com/images/M/meta-icon-new-facebook-2021-logo-83520C311D-seeklogo.com.png height=20px alt="Meta"></a> and  <a href=https://huggingface.co/spaces/tgieruc/agnostic-segmentation)>Agnostic segmentation <img src=https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png height=20px alt="HuggingFace"/></a>
* Visualization of the results

### :fire: Caption Generation
<p align="center">
  <img src="./docs/images/screenshots/captioning.png" alt="caption"/>
</p>

### :fire: Translation
<p align="center">
  <img src="./docs/images/screenshots/translating.png" alt="translation"/>
</p>

### :fire: Preprocessing
<p align="center">
  <img src="./docs/images/screenshots/preprocessing.png" alt="preprocessing"/>
</p>

### :fire: Phrase Grounding
<p align="center">
  <img src="./docs/images/screenshots/phrase_grounding.png" alt="phrase grounding"/>
</p>

### :fire: Segmentation
<p align="center">
  <img src="./docs/images/screenshots/segmentation.png" alt="segmentation"/>
</p>

### :fire: Visualization
<p align="center">
  <img src="./docs/images/screenshots/visualization.png" alt="visualization"/>
</p>



## Installation

* Clone the repository and its submodules
```bash
git clone --recurse-submodules https://github.com/tgieruc/Heritage-in-the-digital-age.git
```

* Install the dependencies
```bash
bash setup.sh
```

* Run the server
```bash
python3 webui.py
```

## Acknowledgements

Thanks to the following people for their help and their work:
* The caption generation pipeline: [Chenkai Wang](https://github.com/cnWangChenkai)
* The English to French translation model: [MarianMT](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en)
* Captioning: [LAVIS](https://github.com/salesforce/LAVIS)
* Phrase Grounding: [GLIP](https://github.com/microsoft/GLIP), [MDETR](https://github.com/ashkamath/mdetr), [groundingDINO](https://github.com/IDEA-Research/GroundingDINO)
* The NLP model for ranking the expressions: [DistilBERT](https://arxiv.org/abs/1910.01108)
* One segmentation model was created using the [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch) library
* The other segmentation models from [Segment Anything](https://github.com/facebookresearch/segment-anything)



## Contact

You can reach me [here](mailto:theo.gieruc@gmail.com?subject=%5BHeritage%20in%20the%20Digital%20Age%5D) 😊
