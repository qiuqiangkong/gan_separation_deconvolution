# Single-Channel Signal Separation and Deconvolution with Generative Adversarial Networks

This codebase is the implementation of the paper Single-Channel Signal Separation and Deconvolution with Generative Adversarial Networks. The paper is accepted by IJCAI 2019. 

## Environments

Python 3.7

Pytorch 1.0

## Run the code

First, set your workspace. 

```
WORKSPACE='/vol/vssp/msos/qk/workspaces/gan_source_separation'
```
Second, train DCGAN model. 

```
CUDA_VISIBLE_DEVICES=0 python pytorch/main.py train --workspace=$WORKSPACE --cuda
```

Third, do denoising, impainting, completition

```
TASK='denoise'    # 'denoise' | 'impaint' | 'complete'
CUDA_VISIBLE_DEVICES=0 python pytorch/denoise_impaint_complete.py --workspace=$WORKSPACE --task=$TASK --repeats_num=8 --cuda
```

Fourth, do source separation and deconvolution

```
CUDA_VISIBLE_DEVICES=0 python pytorch/deconvolve_separate.py --workspace=$WORKSPACE --task=deconvolve_separate --repeats_num=8 --cuda
```

## Results
Denoising, impainting and completion results:
<img src="appendixes/denoising_impainting_completion.png" width="600">

Separation and deconvolution results:
<img src="appendixes/separation_deconvolution.png" width="450">

## Citation

To appear. 