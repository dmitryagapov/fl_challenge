FROM tensorflow/tensorflow:latest
ARG EPOCHS=3
ARG STEPS_PER_EPOCH=1000
RUN apt update && \
    apt install -y cmake

COPY src ./src
WORKDIR src
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV EPOCHS=${EPOCHS}
ENV STEPS_PER_EPOCH=${STEPS_PER_EPOCH}

CMD python load_data.py && \
    horovodrun -np 3 -H localhost:3 python train.py --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH

