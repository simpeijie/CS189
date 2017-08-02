# Digit Recognition

A neural network to classify handwritten digits using raw pixels as features. 

## Implementation

The neural network is implemented using the `numpy` and `scipy' libraries. 

To get the training and test data, run

```bash
./get_data.sh
```

The neural network consists of the input and output layer as well as one hidden layer. The hidden layer uses the ReLU activation function, whereas the output layer uses the softmax as the activation function. The neural net is trained using cross-entropy loss.

The MNIST dataset contains 60k labeled samples, where it is split into a 50k training set and 10k validation set. With one hidden layer, this implementation was able to achieve a training accuracy of 99% and a test accuracy of over 97%. 