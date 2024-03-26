FROM pytorch:23.02-py3

# Umgebungsvariablen setzen
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=0,1,2,3
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV PATH=/usr/local/cuda/bin:$PATH

# Keine GUI-Interaktion waehrend der Installation: 
ARG DEBIAN_FRONTEND=noninteractive

# Working Directory
WORKDIR /workspace

# Add necessary libraries here:
RUN pip install --upgrade pip
RUN pip install wandb
RUN pip install pytest
RUN pip install pytest-cov
RUN pip install torchmetrics
RUN pip install lightning
RUN pip install git+https://github.com/qubvel/segmentation_models.pytorch
# RUN pip install git+https://github.com/Lightning-AI/metrics.git@release/stable