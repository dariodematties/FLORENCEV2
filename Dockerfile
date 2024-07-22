# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:24.06-py3

#RUN apt-get update && apt-get install -y \
    #ffmpeg libsm6 libxext6 libhdf5-dev

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
## see this for torch installation https://pytorch.org/get-started/previous-versions/#v1131
#RUN pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu116
#RUN pip install pywaggle[all]
#RUN pip install h5py
## fix numpy compatibility issue
#RUN pip install "numpy<2"
## native pyymal version is too old with the python
## RUN pip install --ignore-installed PyYAML

## COPY . .

#CMD ["/bin/sh", "-c", "bash"]

