# Verisons of pytorch and cuda taken from conda installation instructions in README.md
# conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y

from pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime 

RUN apt-get update && apt-get install -y build-essential ninja-build
RUN pip install fvcore iopath --no-input
# RUN pip install bottler nvidiacub --no-input
# RUN pip install pytorch3d=0.7.4 pytorch3d --no-input
RUN pip install pyg pyg --no-input

RUN pip install Cython --no-input

COPY . /workspace
WORKDIR lib_shape_prior

RUN python setup.py build_ext --inplace
RUN pip install -U python-pycg[all] -f https://pycg.huangjh.tech/packages/index.html

WORKDIR /workspace
RUN pip install -r requirements.txt