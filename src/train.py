import argparse

import horovod.tensorflow.keras as hvd
import tensorflow as tf
from tensorflow.keras import datasets, models, layers


def define_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='softmax'))

    scaled_lr = 0.001 * hvd.size()
    opt = tf.optimizers.Adam(scaled_lr)

    opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1, average_aggregated_gradients=True)

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    return model


def get_train_dataset(hvd_rank, hvd_size):
    (train_images, train_labels), _ = datasets.cifar10.load_data()
    ds_size = len(train_images)

    start = int(hvd_rank * (ds_size / hvd_size))
    stop = int(start + ds_size / hvd_size)

    dataset = tf.data.Dataset.from_tensor_slices(
        (train_images[start:stop] / 255.0,
         train_labels[start:stop])
    )
    dataset = dataset.repeat().shuffle(10000).batch(128)
    return dataset


def get_test_dataset():
    _, (test_images, test_labels) = datasets.cifar10.load_data()
    return test_images / 255.0, test_labels


def train(model, dataset, epochs, steps_per_epoch, hvd_rank=0, hvd_size=1):
    scaled_lr = 0.001 * hvd.size()
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restore dfrom a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=1, verbose=1),
    ]

    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    # Horovod: write logs on worker 0.
    verbose = 1 if hvd_rank == 0 else 0
    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch // hvd_size, callbacks=callbacks, verbose=verbose,
              validation_data=get_test_dataset())
    return model


def parse_args():
    parser = argparse.ArgumentParser(prog='Distributed training')
    parser.add_argument('--epochs', metavar='EPOCHS', type=int,
                        help='Integer. Number of epochs to train the model.', default=3)
    parser.add_argument('--steps_per_epoch', metavar='STEPS', type=int,
                        help='Total number of steps (batches of samples)', default=1000)
    return parser.parse_args()


def main(epochs, steps_per_epoch, hvd_rank=0, hvd_size=1):
    model = define_model()
    dataset = get_train_dataset(hvd_rank=hvd_rank, hvd_size=hvd_size)
    trained_model = train(model, dataset, epochs, steps_per_epoch, hvd_rank, hvd_size)
    test_images, test_labels = get_test_dataset()
    trained_model.evaluate(test_images, test_labels, verbose=1)
    if hvd_rank == 0:
        trained_model.save('result')


if __name__ == '__main__':
    args = parse_args()
    hvd.init()
    main(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        hvd_rank=hvd.rank(),
        hvd_size=hvd.size(),
    )
