# RUN

### Clone repo

```
git clone https://github.com/dmitryagapov/fl_challenge
```

### Go to repo's folder

```
cd fl_challenge
```

### Build image
default amount of epochs=3, steps_per_epoch=1000, if you want to change it go to [Build with Arguments](README.md#build-with-arguments)

```
docker build -t da/fl_challenge .
```

### Run container

```
docker run da/fl_challenge
```

# ADVANCED
### BUILD WITH ARGUMENTS

You can change the default amount of epochs and steps_per_epoch
```
docker build --build-arg EPOCHS=5 --build-arg STEPS_PER_EPOCH=3000 -t da/fl_challenge .
```

### RUN WITH PARAMS
You can change default amount of epochs and steps_per_epoch or that you specified in [Build with Arguments](README.md#build-with-arguments)
```
docker run -e STEPS_PER_EPOCH=10000 -e EPOCHS=30 da/fl_challenge
```

## Approach

1. Before training load full dataset
2. Using `horovod` to run distributed training
3. Prepare a dataset for every worker `dataset_size = dataset_size_full / amount_of_workers`
4. Change `steps_per_epoch = steps_per_epoch_required / amount_of_workers` 


