FROM nvcr.io/nvidia/pytorch:23.03-py3
RUN apt-get update
RUN apt-get install -y openssh-server && mkdir -p /run/sshd
RUN pip install -U openmim
RUN pip install opencv-python==4.6.0.66
RUN pip install timm pytorch_metric_learning yapf addict
RUN mim install mmengine==0.8.4
RUN mim install mmpretrain==1.0.2
RUN mim install mmdet==3.1.0
RUN pip install mmsegmentation==1.1.1
