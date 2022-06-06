#!/bin/bash

python=~/anaconda3/envs/py36/bin/python

gpu_index=3
root_path="/home/datasets/generative"
model=style_GAN_v2_G
# generate_dataset=style_GAN_init


$python generate.py --gpu_index $gpu_index --root_path $root_path --model $model  --generate_dataset $generate_dataset \
--batch_size 100 --sample 200 --noise_num 10 --image_resolution 32  --style_gan_style_dim 64  --style_gan_n_mlp 1  --style_gan_cmul 1

# $python generate.py --gpu_index $gpu_index --root_path $root_path --model $model  --generate_dataset $generate_dataset \
# --batch_size 100 --sample 5000 --noise_num 10 --image_resolution 32  --style_gan_style_dim 64  --style_gan_n_mlp 1  --style_gan_cmul 1

# generate_dataset=style_GAN_init_64
# # $python generate.py --gpu_index $gpu_index --root_path $root_path --model $model  --generate_dataset $generate_dataset \
# # --batch_size 100 --sample 200 --noise_num 10 --image_resolution 64  --style_gan_style_dim 64  --style_gan_n_mlp 1  --style_gan_cmul 1

# $python generate.py --gpu_index $gpu_index --root_path $root_path --model $model  --generate_dataset $generate_dataset \
# --batch_size 100 --sample 5000 --noise_num 10 --image_resolution 64  --style_gan_style_dim 64  --style_gan_n_mlp 1  --style_gan_cmul 1



# generate_dataset=style_GAN_init_64_c200

# $python generate.py --gpu_index $gpu_index --root_path $root_path --model $model  --generate_dataset $generate_dataset \
# --batch_size 50 --sample 250 --noise_num 200 --image_resolution 64  --style_gan_style_dim 256  --style_gan_n_mlp 1  --style_gan_cmul 1 \
# --style_gan_sample_z_mean 0.5  --style_gan_sample_z_std 0.15

# model=cifar_conv_decoder
# generate_dataset=cifar_conv_decoder

# $python generate.py --gpu_index $gpu_index --root_path $root_path --model $model  --generate_dataset $generate_dataset \
# --batch_size 100 --sample 200 --noise_num 10 --image_resolution 32  --vae_decoder_z_dim 8 --vae_decoder_ngf 64

# $python generate.py --gpu_index $gpu_index --root_path $root_path --model $model  --generate_dataset $generate_dataset \
# --batch_size 100 --sample 5000 --noise_num 10 --image_resolution 32  --vae_decoder_z_dim 32 --vae_decoder_ngf 64

# generate_dataset=style_GAN_init_32_c100

# $python generate.py --gpu_index $gpu_index --root_path $root_path --model $model  --generate_dataset $generate_dataset \
# --batch_size 100 --sample 500 --noise_num 100 --image_resolution 32  --style_gan_style_dim 256  --style_gan_n_mlp 1  --style_gan_cmul 1 \
# --style_gan_sample_z_mean 0.5  --style_gan_sample_z_std 0.2

# model=Gaussian_Noise
# generate_dataset=Gaussian_Noise
# # $python generate.py --gpu_index $gpu_index --root_path $root_path --model $model  --generate_dataset $generate_dataset \
# # --batch_size 100 --sample 200 --noise_num 10 --image_resolution 32  --vae_decoder_z_dim 8 --vae_decoder_ngf 64
# $python generate.py --gpu_index $gpu_index --root_path $root_path --model $model  --generate_dataset $generate_dataset \
# --batch_size 100 --sample 5000 --noise_num 10 --image_resolution 32  --vae_decoder_z_dim 8 --vae_decoder_ngf 64




# generate_dataset=style_GAN_init_32_c62

# $python generate.py --gpu_index $gpu_index --root_path $root_path --model $model  --generate_dataset $generate_dataset \
# --batch_size 100 --sample 500 --noise_num 62 --image_resolution 32  --style_gan_style_dim 256  --style_gan_n_mlp 1  --style_gan_cmul 1 \
# --style_gan_sample_z_mean 0.5  --style_gan_sample_z_std 0.2


