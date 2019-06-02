#!/bin/bash
# You need to modify this path
WORKSPACE='/vol/vssp/msos/qk/workspaces/gan_source_separation'

# Train DCGAN model
CUDA_VISIBLE_DEVICES=0 python pytorch/main.py train --workspace=$WORKSPACE --cuda

# Denoising, impainting or completion
TASK='denoise'    # 'denoise' | 'impaint' | 'complete'
CUDA_VISIBLE_DEVICES=0 python pytorch/denoise_impaint_complete.py --workspace=$WORKSPACE --task=$TASK --repeats_num=8 --cuda

# Separation and deconvolution
CUDA_VISIBLE_DEVICES=0 python pytorch/deconvolve_separate.py --workspace=$WORKSPACE --task=deconvolve_separate --repeats_num=8 --cuda
