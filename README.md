# alma-dnn

Best use docker:

```bash
./docker/run.sh
```

## Times

- CPU (add initial loading 20s time to first 37s):
    ```bash
    Epoch 1/10
    1000/1000 [==============================] - 37s 36ms/step - loss: 256.7993 - mean_squared_error: 256.7993 - auc: 0.5477  
    Epoch 2/10
    1000/1000 [==============================] - 37s 37ms/step - loss: 87.2902 - mean_squared_error: 87.2902 - auc: 0.6495 
    Epoch 3/10
    1000/1000 [==============================] - 37s 37ms/step - loss: 47.6786 - mean_squared_error: 47.6786 - auc: 0.7192
    Epoch 4/10
    1000/1000 [==============================] - 37s 37ms/step - loss: 29.7242 - mean_squared_error: 29.7242 - auc: 0.7134
    Epoch 5/10
    1000/1000 [==============================] - 36s 36ms/step - loss: 19.8291 - mean_squared_error: 19.8291 - auc: 0.7192
    Epoch 6/10
    1000/1000 [==============================] - 36s 36ms/step - loss: 15.2807 - mean_squared_error: 15.2807 - auc: 0.7442
    Epoch 7/10
    1000/1000 [==============================] - 37s 37ms/step - loss: 11.6482 - mean_squared_error: 11.6482 - auc: 0.7803
    Epoch 8/10
    1000/1000 [==============================] - 36s 36ms/step - loss: 9.3353 - mean_squared_error: 9.3353 - auc: 0.8138 
    Epoch 9/10
    1000/1000 [==============================] - 37s 37ms/step - loss: 7.7828 - mean_squared_error: 7.7828 - auc: 0.7942 
    Epoch 10/10
    1000/1000 [==============================] - 36s 36ms/step - loss: 6.6411 - mean_squared_error: 6.6411 - auc: 0.8221
    ```

## help

For datasets that do not fit into memory, there is an answer in the [Keras Documentation FAQ section](https://keras.io/getting-started/faq/#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory)

> You can do batch training using `model.train_on_batch(X, y)` and `model.test_on_batch(X, y)`. See the [models documentation](https://keras.io/models/sequential).
>
> Alternatively, you can write a generator that yields batches of training data and use the method `model.fit_generator(data_generator, samples_per_epoch, nb_epoch)`.
>
> You can see batch training in action in our [CIFAR10 example](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py).

So if you want to iterate your dataset the way you are doing, you should probably use `model.train_on_batch` and take care of the batch sizes and iteration yourself.

One more thing to note is that you should make sure the order in which the samples you train your model with is shuffled after each epoch. The way you have written the example code seems to not shuffle the dataset. You can read a bit more about shuffling [here](https://stackoverflow.com/questions/8101925/effects-of-randomizing-the-order-of-inputs-to-a-neural-network) and [here](https://www.quora.com/Does-the-order-of-training-data-matter-when-training-neural-networks)
