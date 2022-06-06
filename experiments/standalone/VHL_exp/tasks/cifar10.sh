export dataset=${dataset:-"cifar10"}
export model_input_channels=${model_input_channels:-3}
export model=${model:-"resnet18_v2"}
export lr=${lr:-0.01}
export image_resolution=32
export VHL_dataset_list=${VHL_dataset_list:-"['style_GAN_init']"}
export VHL_dataset_batch_size=${VHL_dataset_batch_size:-128}