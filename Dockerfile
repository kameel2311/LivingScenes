# Verisons of pytorch and cuda taken from conda installation instructions in README.md
# conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y

# from pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime 

# RUN apt-get update && apt-get install -y build-essential ninja-build
# RUN pip install fvcore iopath --no-input
# # RUN pip install bottler nvidiacub --no-input
# # RUN pip install pytorch3d=0.7.4 pytorch3d --no-input
# RUN pip install pyg pyg --no-input

# RUN pip install Cython --no-input

# COPY . /workspace
# WORKDIR lib_shape_prior

# RUN python setup.py build_ext --inplace
# RUN pip install -U python-pycg[all] -f https://pycg.huangjh.tech/packages/index.html

# WORKDIR /workspace
# RUN pip install -r requirements.txt

from pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime 

RUN apt-get update && apt-get install -y build-essential ninja-build
RUN pip install fvcore iopath pyg Cython --no-input && \
    apt-get install ffmpeg libsm6 libxext6 libsm6 libxext6 libxrender-dev -y

# RUN conda create -n livingscenes python=3.9 -y && \
#     conda activate livingscenes && \ 
#     conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y && \ 
#     conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y && \ 
#     conda install -c bottler nvidiacub -y && \
#     conda install pytorch3d=0.7.4 -c pytorch3d -y && \
#     conda install pyg -c pyg -y

RUN conda create -n livingscenes python=3.9 -y && \ 
    echo "source activate livingscenes" > ~/.bashrc && \
    # Install PyTorch with CUDA support
    conda install -n livingscenes pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y && \
    # Install fvcore and iopath
    conda install -n livingscenes -c fvcore -c iopath -c conda-forge fvcore iopath -y && \
    # Install NVIDIA CUB
    conda install -n livingscenes -c bottler nvidiacub -y && \
    # Install PyTorch3D
    conda install -n livingscenes pytorch3d=0.7.4 -c pytorch3d -y && \
    # Install PyTorch Geometric
    conda install -n livingscenes pyg -c pyg -y

COPY . /workspace
WORKDIR lib_shape_prior

RUN python setup.py build_ext --inplace
RUN pip install -U python-pycg[all] -f https://pycg.huangjh.tech/packages/index.html && \
    pip install open3d

WORKDIR /workspace
RUN pip install -r requirements.txt