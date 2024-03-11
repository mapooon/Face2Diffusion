# Face2Diffusion (CVPR2024)
<a href='https://arxiv.org/abs/2403.05094'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; 
<a href='https://mapooon.github.io/Face2DiffusionPage'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; 
![Overview](fig/teaser.png)
The official PyTorch implementation for the following paper:
> [**Face2Diffusion for Fast and Editable Face Personalization**](https://arxiv.org/abs/2403.05094),  
> Kaede Shiohara, Toshihiko Yamasaki,   
> *CVPR 2024*

# Changelog
2024/03/11: Released this repository and demo code.

# Recomended Development Environment
* GPU: NVIDIA A100
* CUDA: 12.2
* Docker: 535.129.03


# Setup
## 1. Build Docker Image
Build a docker image
```bash
docker build -t face2diffusion dockerfiles/
```
Execute docker using the following command (Please replace `/path/to/this/repository` with a proper one)
```bash
docker run -it --gpus all --shm-size 512G \
-v /path/to/this/repository:/workspace \
face2diffusion bash
```
Install some packages
```bash
bash install.sh
```
Overwrite a file for Face2Diffusion's pipeline
```bash
cp /workspace/src/modified_scripts/modeling_clip.py /opt/conda/lib/python3.10/site-packages/transformers/models/clip/
```

## 2. Download checkpoints
We provide checkpoints for [mapping network](https://drive.google.com/file/d/1Lf_mwMgme_HVYJCkViGr4TfGOfKw9PhE/view?usp=sharing) and MSID encoder (Coming soon!) and place them to ```checkpoints/```.

# Demo
Currently, we provide pre-computed identity features from the MSID encoder. The extracted features and the original images can be seen in `./input/` folder. Here is an example to generate images:
```bash
python3 inference_f2d.py \
-w checkpoints/mapping.pt \
-i input/0.npy \ # input identity
-p 'f l eating bread in front of the Eiffel Tower' \ # input prompt
-o output.png \ # output file name
-n 8 \ # num of images to generate
```
Note: The identifier S* should be represented as "f l".

Full pipeline is coming soon!

# Citation
If you find our work useful for your research, please consider citing our paper:
```bibtex
@inproceedings{shiohara2024face2diffusion,
  title={Face2Diffusion for Fast and Editable Face Personalization},
  author={Shiohara, Kaede and Yamasaki, Toshihiko},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```