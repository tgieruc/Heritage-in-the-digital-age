#!/bin/bash

# DINO
HOME=$PWD

echo "Downloading DINO model..."

mkdir -p $HOME/submodules/GroundingDINO/model
wget -q --show-progress  -nc https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O $PWD/submodules/GroundingDINO/model/groundingdino_swint_ogc.pth

cd $HOME/submodules/GroundingDINO

echo 
echo "Installing DINO requirements..."

pip install -q -e .

cd $HOME

echo 
echo "Installing Segment Anything..."
# Segment Anything
pip install -q git+https://github.com/facebookresearch/segment-anything.git

echo 
echo "Installing AgnosticSegment..."
pip install --upgrade gdown

mkdir -p $HOME/model
# check if model is already downloaded
if [ ! -f "$HOME/model/agnostic_segment.pth" ]; then
    echo "Downloading AgnosticSegment model..."
    gdown https://drive.google.com/uc?id=1OWH7arM-qllbCJwqkMVy9NKWHB398iol -O $HOME/model/agnostic_segment.pth
fi

echo
echo "Installing last requirements..."
pip install -q -r requirements.txt

