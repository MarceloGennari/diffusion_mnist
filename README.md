# Diffusion Model on MNIST dataset
This is a simple and minimal implementation of DDPM on the MNIST dataset
Many of the details from the original paper are missing, but the image generation is working fine

## Training UNet
To train the UNet, you just need to run 
`python main.py`

## Inference
After training the UNet, you can get some samples by running
`python inference_unet.py`
