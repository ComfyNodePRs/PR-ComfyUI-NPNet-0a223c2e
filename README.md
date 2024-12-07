ComfyUI NPNet

A very barebones copypaste implementation of https://github.com/xie-lab-ml/Golden-Noise-for-Diffusion-Models

Note: *only* works with square noise, apparently. If you pass in other shaped latents, it will print an error and use the unmodified noise.


Use with custom sampling and pass in an initial noise from eg. `RandomNoise` and a cond (only the first prompt in the conditioning will be used if multiple exist).
