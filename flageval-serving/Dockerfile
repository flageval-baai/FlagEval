FROM nvcr.io/nvidia/pytorch:23.03-py3
RUN apt-get update
RUN apt-get install -y openssh-server && mkdir -p /run/sshd

RUN pip install transformer-engine==0.6.0 transformers==4.31.0 transformers-stream-generator==0.0.4

RUN pip install sentencepiece==0.1.98 accelerate==0.21.0 colorama==0.4.6 cpm_kernels==1.0.11 streamlit==1.25.0 fairscale==0.4.13

# Qwen
RUN pip install tiktoken