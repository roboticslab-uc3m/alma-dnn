# alma-dnn

For datasets that do not fit into memory, there is an answer in the [Keras Documentation FAQ section](https://keras.io/getting-started/faq/#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory)

> You can do batch training using `model.train_on_batch(X, y)` and `model.test_on_batch(X, y)`. See the [models documentation](https://keras.io/models/sequential).
>
> Alternatively, you can write a generator that yields batches of training data and use the method `model.fit_generator(data_generator, samples_per_epoch, nb_epoch)`.
>
> You can see batch training in action in our [CIFAR10 example](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py).

So if you want to iterate your dataset the way you are doing, you should probably use `model.train_on_batch` and take care of the batch sizes and iteration yourself.

One more thing to note is that you should make sure the order in which the samples you train your model with is shuffled after each epoch. The way you have written the example code seems to not shuffle the dataset. You can read a bit more about shuffling [here](https://stackoverflow.com/questions/8101925/effects-of-randomizing-the-order-of-inputs-to-a-neural-network) and [here](https://www.quora.com/Does-the-order-of-training-data-matter-when-training-neural-networks)