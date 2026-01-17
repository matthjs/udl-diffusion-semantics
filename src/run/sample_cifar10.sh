#!/bin/sh

source .venv/bin/activate
source ./src/run/set_pythonpath.sh

python3 ./src/scripts/sample_vqvae.py\
    --dataset="cifar10"\
    --save_dir="./generations"\
    --model_params="./pretrained/vqvae-cifar10.pth"\
    --batch_size=100\
    --n_samples=10\
    --temp=1\
    --n_embeds=128\
    --hidden_dim=64\
    --n_pixelcnn_res_blocks=2\
    --n_pixelcnn_conv_blocks=2\
