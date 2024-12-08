# ComfyUI NPNet (Golden Noise)

A very barebones mostly-copypaste implementation of https://github.com/xie-lab-ml/Golden-Noise-for-Diffusion-Models

## Requirements
You need the pre-trained weights for your model from https://drive.google.com/drive/folders/1Z0wg4HADhpgrztyT3eWijPbJJN5Y2jQt?usp=drive_link

## Usage
Use with custom sampling and pass in an initial noise from eg. `RandomNoise` and a cond (only the first prompt in the conditioning will be used if multiple exist).

You can also run it on the CPU, though appears to change the output for some reason.

## Notes
The model works with 128x128 latents, apparently. If you pass in other shaped latents, it will reshape the noise into a square before running the noise model, and then reshape the result back to the original resolution.

If you get an error from the timm module when running this, update your timm package. It may be too old.
