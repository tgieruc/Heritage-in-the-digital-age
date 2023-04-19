#!/bin/bash

# DINO
HOME=$PWD

mkdir -p $HOME/submodules/GroundingDINO/model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O $PWD/submodules/GroundingDINO/model/groundingdino_swint_ogc.pth

cd $HOME/submodules/GroundingDINO

pip install -q -e .

cd $HOME
# Segment Anything
pip install -q git+https://github.com/facebookresearch/segment-anything.git

pip install -q -r requirements.txt