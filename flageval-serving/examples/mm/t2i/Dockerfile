FROM nvcr.io/nvidia/pytorch:23.03-py3

# Set environment variables.
ENV https_proxy=http://10.1.0.34:7890 \
http_proxy=http://10.1.0.34:7890 \
all_proxy=socks5://10.1.0.34:7891

RUN pip install git+https://github.com/openai/CLIP.git  

COPY t2i_env/stable-diffusion.tar /root 
RUN cd /root && tar -xvf stable-diffusion.tar && rm stable-diffusion.tar && \
cd stable-diffusion && pip install -e .

RUN pip install transformers diffusers invisible-watermark
RUN pip install pytorch_fid pytorch_lightning stable_diffusion taming-transformers-rom1504 kornia 
COPY t2i_env/en_core_web_sm-3.6.0.tar.gz /root
RUN pip install /root/en_core_web_sm-3.6.0.tar.gz
RUN pip install retrying

COPY t2i_env/models--openai--clip-vit-large-patch14.tar /tmp 
RUN tar -xvf /tmp/models--openai--clip-vit-large-patch14.tar -C /tmp && \
rm /tmp/models--openai--clip-vit-large-patch14.tar && \
mkdir -p /root/.cache/huggingface/hub && \
rm -rf /root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14 && \
mv /tmp/models--openai--clip-vit-large-patch14 /root/.cache/huggingface/hub/

ENV https_proxy= \
    http_proxy= \
    all_proxy=

