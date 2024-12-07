ComfyUI NPNet

A very barebones copypaste implementation of https://github.com/xie-lab-ml/Golden-Noise-for-Diffusion-Models

Use with custom sampling and pass in an initial noise from eg. `RandomNoise` and a cond (only the first prompt in the conditioning will be used if multiple exist).

Note: *only* works with 128x128 square noise, apparently. If you pass in other shaped latents, it will reshape the noise into a square before running the noise model, and then reshape the result back to the original resolution.

You can also run it on the CPU, though appears to change the output for some reason.

If you get an error from the timm module when running this, update your timm package. It may be too old.
