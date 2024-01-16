FROM nvcr.io/nvidia/pytorch:23.03-py3
RUN pip install ftfy nltk sentencepiece easydict 

ENV https_proxy=http://10.1.0.34:7890 \
    http_proxy=http://10.1.0.34:7890 \
    all_proxy=socks5://10.1.0.34:7891

RUN pip install git+https://github.com/openai/CLIP.git  

ENV https_proxy= \
    http_proxy= \
    all_proxy=

RUN pip install ruamel.yaml 
RUN pip install transformers==4.11 
RUN pip install timm retrying

# ENV https_proxy=http://10.1.0.34:7890 \
#     http_proxy=http://10.1.0.34:7890 \
#     all_proxy=socks5://10.1.0.34:7891

# RUN pip install git+https://github.com/salesforce/ALBEF.git

# ENV https_proxy= \
#     http_proxy= \
#     all_proxy=
