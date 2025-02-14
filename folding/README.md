# alma-dnn/folding

Train with labeled data generated via <https://github.com/roboticslab-uc3m/alma-dataset>. You can also directly train with the publicly available dataset <https://doi.org/10.5281/zenodo.14864392>.

## cloth2d_dnn.py

![assets/cloth2d_dnn_keras.png](assets/cloth2d_dnn_keras.png)

### Build

Best build via docker:

```bash
./docker/build.sh
```

### Run

Best run via docker:

```bash
./docker/run.sh
```

### Example Output

#### Times

CPU (add initial loading 20s time to first 37s):

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

#### Accuracy

Displaying [pick_x pick_y place_x place_y].

- `dataset/image10000.png`

```
ground truth [58. 47. 89. 48.]
DNN [54.942677  45.316486  86.94711   48.10437  ]
```

- `dataset/image10001.png`

```
ground truth [52. 50.  9. 59.]
DNN [52.015865  46.120808   7.902022  54.8517533 ]
```

- `dataset/image10004.png`

```
ground truth [41. 20. 36.  9.]
DNN [37.37261   16.569086  34.050407   9.787606 ]
```
