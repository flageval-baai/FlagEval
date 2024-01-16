FROM nvcr.io/nvidia/pytorch:23.03-py3
ENV https_proxy=http://10.1.0.34:7890 \
http_proxy=http://10.1.0.34:7890 \
all_proxy=socks5://10.1.0.34:7891

RUN pip install salesforce-lavis scipy nltk easydict num2words retrying
COPY bert-base-uncased.tar /tmp 
RUN tar -xvf /tmp/bert-base-uncased.tar -C /tmp && \
mkdir -p /root/.cache/huggingface/hub && \
rm -rf /root/.cache/huggingface/hub/models--bert-base-uncased && \
mv /tmp/models--bert-base-uncased /root/.cache/huggingface/hub/
# nltk-downloaded
RUN wget -d https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip && \
    mkdir -p /root/nltk_data/corpora && \
    cp wordnet.zip /root/nltk_data/corpora  

ENV https_proxy= \
http_proxy= \
all_proxy=