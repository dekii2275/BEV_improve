FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV FORCE_CUDA=1

# Fix NVIDIA GPG key issue and install system dependencies
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    git wget ninja-build g++ \
    && rm -rf /var/lib/apt/lists/*

# Install other requirements first
RUN pip install --no-cache-dir \
    numba==0.48.0 \
    numpy==1.19.5 \
    nuscenes-devkit \
    opencv-python-headless \
    pandas \
    pytorch-lightning==1.5.10 \
    scikit-image \
    scipy \
    setuptools==59.5.0 \
    tensorboardX \
    pyquaternion

# Install mmcv-full (prebuilt for torch1.10 + cuda11.3)
RUN pip install --no-cache-dir \
    mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Install mmdet (no CUDA extensions, fast)
RUN pip install --no-cache-dir mmdet==2.19.0 --no-deps

# Install mmdet3d from source (avoids slow pip dependency resolution)
RUN cd /tmp && \
    git clone https://github.com/open-mmlab/mmdetection3d.git -b v0.18.1 --depth 1 && \
    cd mmdetection3d && \
    pip install --no-cache-dir -e . --no-deps && \
    cd / && rm -rf /tmp/mmdetection3d

WORKDIR /workspace/BEVHeight
