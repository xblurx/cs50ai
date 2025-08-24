1. Start from the same layers config as in the lecture for CNNs,
but changing the input shape channel value to 3 as the images in the dataset is rgb.  
That produced poor results - 0.06 acc without increasing after 3 epochs with 3.5 loss

2. then i tried increasing the number of filters from 32 to 64 and increasing number of hidden nodes from 128 to 256 but it was not a success.

3. I noticed that i did not load the last category of images, fixed it. Removed output layer activation function. tested different learning rates, sparse_categorical_crossentropy loss function, removed CNN layers, all for nothing. Then I switched dropout rate from .5 to .2 and it started working but produced very random results, going up to 0.94 accuracy, but sometimes staying at 0.15. maybe its cause of shuffling the dataset, idk.

4. after adding static random_state when splitting the data into train/test axises i noticed improvements, the model was learning up to 92% acc more often, but still unstable

5. after the small modification of loading the data - which just consisted of returning the np.array from load_data and more importantly - normalizing the images array the learning went stable with the acc of 99%. then i suspected the overfitting and increased the dropout back to 0.5 and accuracy was up to 97%.

6. i decided to go further, and investigated how to improve overall stability, feature extraction. i tried different various layer setups, ended up with 2 sets of 2 conv, normalize, pool, dropout layers before dense layer, and the metrics improved even better.
