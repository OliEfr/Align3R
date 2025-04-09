FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and other necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    python3.11-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    unzip \
    tar \
    htop \
    qt5-qmake \
    qtbase5-dev \
    xvfb \
    ffmpeg \
    python3-tk \ 
    ninja-build \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Set Python 3.11 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN echo 'source /opt/venv/bin/activate' >> ~/.bashrc

# Nvidia env vars
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Add user and set as default
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

# Fix permissions for pip installations
RUN mkdir -p /opt/venv/lib/python3.11/site-packages && \
    chown -R ${USERNAME}:${USERNAME} /opt/venv

# Switch to non-root user
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Install PyTorch and dependencies
RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install
COPY --chown=${USERNAME}:${USERNAME} requirements.txt .
COPY --chown=${USERNAME}:${USERNAME} requirements_optional.txt .
RUN pip install -r requirements.txt
RUN pip install -r requirements_optional.txt

COPY --chown=${USERNAME}:${USERNAME} croco/models/curope/ ./croco/models/curope/ 
ENV TORCH_CUDA_ARCH_LIST="8.9"
RUN cd croco/models/curope/ && \
    python setup.py build_ext --inplace && \
    cd ../../../

COPY --chown=${USERNAME}:${USERNAME} third_party/ml-depth-pro ./third_party/ml-depth-pro
RUN cd third_party/ml-depth-pro && \
    pip install -e . && \
    pip install transformers==4.41.2

COPY --chown=${USERNAME}:${USERNAME} third_party/sam2 ./third_party/sam2
RUN cd third_party/sam2 && \
    pip install -e . && \
    cd ../../

# run this before: git clone https://github.com/DepthAnything/PromptDA.git && cd
COPY --chown=${USERNAME}:${USERNAME} PromptDA ./PromptDA
RUN cd PromptDA && \
    pip install -e .

# run this before: git clone https://github.com/Junyi42/viser.git &&
COPY --chown=${USERNAME}:${USERNAME} viser ./viser
RUN cd viser && \
    pip install -e .
