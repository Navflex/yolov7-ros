#!/bin/bash

# Update and install system libraries
echo "Updating and installing system libraries..."
sudo apt-get update
sudo apt-get install -y liblapack-dev libblas-dev gfortran libfreetype6-dev libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev
sudo apt-get install -y python3-pip

# Install Python packages from requirements.txt
echo "Installing Python packages from requirements.txt..."
sudo -H pip3 install -r requirements.txt

# Install PyCUDA
echo "Setting up environment variables for CUDA and installing PyCUDA..."
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
python3 -m pip install pycuda --user

# Install Seaborn
echo "Installing Seaborn..."
sudo apt install -y python3-seaborn

# Install PyTorch and torchvision
echo "Installing PyTorch and torchvision..."
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo -H pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
sudo python3 setup.py install
cd ..

# Optional: Install Jetson Stats
echo "Installing jetson-stats (optional)..."
sudo python3 -m pip install -U jetson-stats==3.1.4

# TensorRT Installation and Verification
echo "Checking for TensorRT installation..."
dpkg-query --show nvidia-jetpack
dpkg -l | grep nvinfer

# Install TensorRT Python bindings if not installed
echo "Installing TensorRT Python bindings..."
sudo apt-get update
sudo apt-get install -y python3-libnvinfer-dev

# Set up TensorRT environment variables
echo "Setting up TensorRT environment variables..."
echo 'export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify TensorRT installation
echo "Verifying TensorRT installation..."
python3 -c "import tensorrt as trt; print(trt.__version__)"

echo "All dependencies, including TensorRT, are installed."
