FROM tensorflow/tensorflow:latest

RUN apt update && \
    apt install -y cmake

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


COPY *.py ./
CMD python load_data.py && \
    horovodrun -np 3 -H localhost:3 python train.py --epochs 3 --steps_per_epoch 1000


