FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN apt-get install ffmpeg libsm6 libxext6  git -y

RUN pip install opencv-python-headless==4.5.5.64 matplotlib seaborn scikit-learn scikit-image kornia==0.6.8 chumpy

RUN pip install omegaconf pytorch-lightning==1.7.0 einops av


RUN pip install transformers==4.19.2
RUN pip install diffusers invisible-watermark
RUN pip install --upgrade transformers

RUN python -m pip install --upgrade pip setuptools

RUN pip install fire wandb safetensors
RUN pip install accelerate==0.16.0 datasets==2.8.0 ftfy==6.1.1 tensorboard==2.11.2 Jinja2==3.1.2
WORKDIR /workspace