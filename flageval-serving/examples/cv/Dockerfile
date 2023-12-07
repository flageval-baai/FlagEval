FROM nvcr.io/nvidia/pytorch:23.03-py3
RUN apt-get update
RUN apt-get install -y openssh-server && mkdir -p /run/sshd
RUN pip install -U openmim
RUN pip install opencv-python==4.6.0.66 timm==0.9.7 pytorch_metric_learning==2.3.0 yapf==0.40.1  einops==0.6.1 ninja  addict
RUN mim install mmengine==0.8.4 mmpretrain==1.0.2  mmdet==3.1.0
RUN pip install mmsegmentation==1.1.1
RUN pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.16#egg=xformers
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

