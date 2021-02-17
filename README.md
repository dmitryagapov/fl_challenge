# RUN

### Clone repo

```
git clone fl_comp
```

### Go to repo's folder

```
cd fl_comp
```

### Build image

```
docker build -t da/fl_comp .
```

### Run container

```
docker run da/fl_comp
```

## Approach

1. Before training load full dataset
2. Using `horovod` run distributed training for `3` workers
3. Prepare dataset for every worker `dataset_size = dataset_size_full / amount_of_workers`
4. Change `steps_per_epoch = steps_per_epoch_required / amount_of_workers` 


