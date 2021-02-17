FROM tensorflow/tensorflow:latest
ARG EPOCHS=3
ARG STEPS_PER_EPOCH=500
ARG WORKERS=3
RUN apt update && \
    apt install -y cmake

COPY src ./src
WORKDIR src
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV EPOCHS=${EPOCHS}
ENV STEPS_PER_EPOCH=${STEPS_PER_EPOCH}
ENV WORKERS=${WORKERS}

CMD python load_data.py && \
    horovodrun -np $WORKERS -H localhost:$WORKERS python train.py --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH

