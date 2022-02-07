#!/bin/bash



with_cuda=0

if [ "$1" == "--cuda" ] || [ "$1" == -c ] ; then
    with_cuda=1
fi

echo "with_cuda=$with_cuda"

if [[ $with_cuda -eq 1 ]] ; then
    DIST_NAME=linux-cuda
else
    DIST_NAME=linux
fi

mkdir -p ./build/resources/

# download resources
echo ""
echo "Downloading model's checkpoint..."
s="https://w3.impa.br/~rodrigo.loro/fetal_mri/amnioml.ckpt"
d="./build/resources/amnioml.ckpt"
wget "$s" -O "$d"

mkdir -p "./build/$DIST_NAME"


echo "Setting up the environment..."
python3 -m venv "./build/$DIST_NAME/venv"

source build/$DIST_NAME/venv/bin/activate

#echo "Updating pip..."
pip install --upgrade pip

# Install torch with CUDA
# Both options are working, CUDA 10.2 was chosen to increase compatibility
# pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

if [[ $with_cuda -eq 1 ]] ; then
    pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio===0.10.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
else
    pip3 install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
fi

pip install pytorch_lightning
pip install pynrrd
pip install pyinstaller

mkdir -p "./build/$DIST_NAME/env_log"
pip freeze > "./build/$DIST_NAME/env_log/$( date -Is )"

# remove previous build files
mkdir -p "./build/$DIST_NAME/tmp"
rm -rf "./build/$DIST_NAME/tmp/*"

# remove previous dist files
mkdir -p "./dist/$DIST_NAME"
rm -rf "./dist/$DIST_NAME/*"

cp -r  "./AmnioML" "./dist/$DIST_NAME"

# create the binary
echo "Building binaries..."
export PYTHONPATH='.'
pyinstaller ./src/models/predict.py --workpath ./build/$DIST_NAME/tmp --distpath ./dist/$DIST_NAME/AmnioML -p '.'  --add-data src:src -w -y

# make the filepath consistent
mv dist/$DIST_NAME/AmnioML/predict/predict dist/$DIST_NAME/AmnioML/predict/predict.exe

echo "Copying resources..."
cp ./build/resources/amnioml.ckpt ./dist/$DIST_NAME/AmnioML/predict


