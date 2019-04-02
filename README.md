# Overview

  A test implementation for the paper "Multi-Granularity Generative Adversarial Nets with Reconstructive Sampling for Image Inpainting" (Under reviewing)

# Environment

  [Torch7](http://torch.ch/docs/getting-started.html)

# Supported Toolkits
  
  [nn](https://github.com/torch/nn)
  
  [image](https://github.com/torch/image)

# Demo
  
  1. Download pre-trained models from [BaiduNetdisk](https://pan.baidu.com/s/1ilBXZUZlACeChzE9w-zRxQ). password: 48mk.

  2. Download test images and mask samples, then put the pre-trained model, test images and masks into same dir

  3. Inpaint the damaged image with designated mask:
     
     th test_for_inpainting.lua --input raw_image_256.jpg --mask mask_256.png

  4. Inpaint the damaged image with random mask:

     th test_for_inpainting.lua --input raw_image_227.jpg

# Notes
- This is developed on a Linux machine running Ubuntu 16.04.

- Use GPU for the high speed computation.

# Reference
[Globally and Locally Consistent Image Completion](https://github.com/satoshiiizuka/siggraph2017_inpainting)

[Context Encoders: Feature Learning by Inpainting](https://github.com/pathak22/context-encoder)

[High-Resolution Image Inpainting Using Multi-Scale Neural Patch Synthesis](https://github.com/leehomyc/Faster-High-Res-Neural-Inpainting)
