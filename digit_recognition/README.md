# Digit Recognition

A neural network to classify handwritten digits using raw pixels as features. 

## Implementation

The neural network is implemented using the `numpy` and `scipy` libraries. 

To get the training and test data, run

```
./get_data.sh
```

## Preprocessing Data

Before implementing the neural network, the labels are one-hot encoded to convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10}. The features are also preprocessed through centering and normalizing. The data is also shuffled to ensure the network is properly trained, i.e. the network does not learn to always output a certain value.

## Structure

The neural network consists of the input and output layer as well as one hidden layer. The hidden layer uses the ReLU activation function, whereas the output layer uses the softmax as the activation function. The neural net is trained using cross-entropy loss.

## Step Size and Convergence

In this implementation, the learning rate is not kept constant. With a learning rate that is too large, training will diverge; too small, training will be slow and stochastic gradient descent will get trapped in a bad local minima. Hence, the learning rate is decayed by a constant factor of 0.9 every epoch. An epoch is one full pass over the data.

## Weights

The weights are initialized from a Gaussian distribution with mean 0 and variance (some constant < 1).

## Validation

The MNIST dataset contains 60k labeled samples, where it is split into a 50k training set and 10k validation set. 