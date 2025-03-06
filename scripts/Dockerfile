FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y \
    sudo

ENV TZ=Europe/London

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# RUN useradd -ms /bin/bash user && \
#     usermod -aG sudo user

# RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# USER user

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive

RUN sudo apt -y install --no-install-recommends \
    automake \
    build-essential \
    cmake \
    curl \
    git \
    python3 \
    python3-dev \
    python3-pip \
    unzip \
    wget \
    zip \
    nano \
    gnupg2

RUN wget -qO - https://ngrok-agent.s3.amazonaws.com/ngrok.asc | apt-key add - \
    && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list \
    && apt-get update && apt-get install -y ngrok


COPY ngrok_token /root/ngrok_token

RUN ngrok config add-authtoken $(cat /root/ngrok_token)

RUN conda init bash

RUN echo "export CUDA_HOME=/usr/local/cuda-11.7" >> /etc/bash.bashrc

ENV CUDA_HOME=/usr/local/cuda-11.7

RUN /opt/conda/bin/python3 -m pip install --upgrade pip

# The important one, specific version of snntorch because they changed the way the rleaky are written
RUN pip3 install git+https://github.com/jeshraghian/snntorch@40ad1360c644f948889cb9610121397c6ce943c8

# To make nni work
RUN python3 -m pip install --no-cache-dir nni==2.7 tqdm "typeguard<3" filelock==3.10 jupyterlab notebook ipywidgets seaborn matplotlib scikit-learn tabulate brevitas==0.10.2 tqdm pandas scipy





RUN
WORKDIR /root
