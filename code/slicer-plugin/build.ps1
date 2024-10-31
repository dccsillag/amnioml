# execute
# Set-Executionpolicy unrestricted -Scope Process
# before running the script

$with_cuda = 0
$DIST_NAME = "win"

if ( ( $args[0] -eq "--cuda" )  -or  ($args[0] -eq "-c" ) ) {
    $with_cuda = 1
    $DIST_NAME = "win-cuda"
}


echo "with_cuda=$with_cuda"
echo "DIST_NAME=$DIST_NAME"



# download resources
echo "Downloading python 3.9.10..."
mkdir -Force ".\build\resources"
$source = "https://www.python.org/ftp/python/3.9.10/python-3.9.10-amd64.exe"
$destination = ".\build\resources\python-3.9.10-amd64.exe"
Invoke-WebRequest -Uri $source -OutFile $destination
echo ""
echo "Downloading model's checkpoint..."
$source = "https://w3.impa.br/~rodrigo.loro/fetal_mri/amnioml.ckpt"
$destination = ".\build\resources\amnioml.ckpt"
Invoke-WebRequest -Uri $source -OutFile $destination
echo ""
echo "Installing Python 3.9..."
Start-Process "./build/resources/python-3.9.10-amd64.exe" -argumentlist "/S" -wait


echo "Adding Python 3.9 to a temporary PATH environment variable..."
$python_path="$( py -3.9 -c "import sys; print(sys.executable[:-10])" )/Scripts"
$env:PATH += ";$python_path"

mkdir -Force ".\build\$DIST_NAME"

echo "Setting up the environment..."
py -3.9 -m venv ".\build\$DIST_NAME\venv"
".\build\$DIST_NAME\venv\Scripts\activate"

# throwing error messages in Windows
#echo "Updating pip..."
#pip install --upgrade pip

# Install torch with CUDA
# Both options are working, CUDA 10.2 was chosen to increase compatibility
# pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
if ( $with_cuda -eq 1 ) {
    pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio===0.10.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
} else {
    pip3 install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
}

pip install pytorch_lightning==1.8.6
pip install pynrrd
pip install pyinstaller

mkdir -Force ".\build\$DIST_NAME\env_log"
$filename=(Get-Date).tostring("yyyy_MM_dd-hh_mm_ss")+".txt"
pip freeze > ".\build\$DIST_NAME\env_log\$filename"

# remove previous build files
mkdir -Force ".\build\$DIST_NAME\tmp"
rm -Recurse -Force ".\build\$DIST_NAME\tmp\*"

# remove previous dist files
mkdir -Force ".\dist\$DIST_NAME"
rm -Recurse -Force ".\dist\$DIST_NAME\*"

cp -Recurse  ".\AmnioML" ".\dist\$DIST_NAME"

# create the binary
echo "Building binaries..."
$env:PYTHONPATH = '.'
pyinstaller ".\src\models\predict.py" --workpath ".\build\$DIST_NAME\tmp" --distpath ".\dist\$DIST_NAME\AmnioML" -w -y

echo "Copying resources..."
cp -Recurse ".\build\resources\amnioml.ckpt" ".\dist\$DIST_NAME\AmnioML\predict\"

