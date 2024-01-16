# Set environment variables.
FROM nvcr.io/nvidia/pytorch:23.06-py3

RUN pip install --upgrade pip -i http://10.1.1.16/repository/pypi-group/simple --trusted-host 10.1.1.16
RUN pip install transformers==4.31.0 tokenizers==0.12.1 \
    sentencepiece==0.1.99 shortuuid accelerate==0.21.0 \
    peft==0.4.0 bitsandbytes==0.41.0 pydantic==1.10.13 \
    markdown2==2.4.10 scikit-learn==1.2.2 gradio==3.35.2 \
    gradio_client==0.2.9 requests httpx==0.24.0 uvicorn fastapi \
    einops==0.6.1 einops-exts==0.0.4 \
    timm==0.6.13 retrying -i http://10.1.1.16/repository/pypi-group/simple --trusted-host 10.1.1.16


COPY LLaVA.tar /root
RUN cd /root && tar -xvf LLaVA.tar && rm LLaVA.tar 

RUN pip install -e /root/LLaVA 

COPY models--openai--clip-vit-large-patch14-336.tar /tmp
RUN mkdir -p /root/.cache/huggingface/hub && \
    tar -xvf /tmp/models--openai--clip-vit-large-patch14-336.tar -C /root/.cache/huggingface/hub && \
    rm -rf /tmp/models--openai--clip-vit-large-patch14-336.tar 
