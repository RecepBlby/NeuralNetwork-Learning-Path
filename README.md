# NeuralNetwork-Learning-Path

# Neural networks
form the base of deep learning, which is a subfield of machine learning, where the structure of the human brain inspires the algorithms. Neural networks take input data, train themselves to recognize patterns found in the data, and then predict the output for a new set of similar data. Therefore, a neural network can be thought of as the functional unit of deep learning, which mimics the behavior of the human brain to solve complex data-driven problems.The first thing that comes to our mind when we think of “neural networks” is biology, and indeed, neural nets are inspired by our brains.

In machine learning, the neurons’ dendrites refer to as input, and the nucleus process the data and forward the calculated output through the axon. In a biological neural network, the width (thickness) of dendrites defines the weight associated with it.

# What is an Artificial Neural Network and why should you use it?
A single perceptron (or neuron) can be imagined as a Logistic Regression. Artificial Neural Network, or ANN, is a group of multiple perceptrons/ neurons at each layer. ANN is also known as a Feed-Forward Neural network because inputs are processed only in the forward direction:
ANN consists of 3 layers – Input, Hidden and Output. The input layer accepts the inputs, the hidden layer processes the inputs, and the output layer produces the result.Each layer tries to learn certain weights

Simply put, an ANN represents interconnected input and output units in which each connection has an associated weight. During the learning phase, the network learns by adjusting these weights in order to be able to predict the correct class for input data.For instance:We encounter ourselves in a deep sleep state, and suddenly our environment starts to tremble. Immediately afterward, our brain recognizes that it is an earthquake. At once, we think of what is most valuable to us:
Our beloved ones.Essential documents.Jewelry.Laptop.A pencil.

What will our priorities be in this case? Perhaps, we are going to save our beloved ones first, and then if time permits, we can think of other things. What we did here is, we assigned a weight to our valuables. Each of the valuables at that moment is an input, and the priorities are the weights we assigned it to it.The same is the case with neural networks. We assign weights to different values and predict the output from them. However, in this case, we do not know the associated weight with each input, so we make an algorithm that will calculate the weights associated with them by processing lots of input data.

Artificial Neural Network is capable of learning any nonlinear function. Hence, these networks are popularly known as Universal Function Approximators. ANNs have the capacity to learn weights that map any input to the output.One of the main reasons behind universal approximation is the activation function. Activation functions introduce nonlinear properties to the network. This helps the network learn any complex relationship between input and output.

# Steps involved in the implementation of a neural network:A neural network executes in 2 steps :
1. Feedforward: we have a set of input features and some random weights. Notice that in this case, we are taking random weights that we will optimize using backward propagation.
2. Backpropagation: we calculate the error between predicted output and target output and then use an algorithm (gradient descent) to update the weight values.
While designing a neural network, first, we need to train a model and assign specific weights to each of those inputs. We assign some random weight to our inputs, and our model calculates the error in prediction. Thereafter, we update our weight values and rerun the code (backpropagation).

# Summarizing an Artificial Neural Network:
Take inputs>>>Add bias (if required)>>>Assign random weights to input features>>Run the code for training.>>Find the error in prediction.>>Update the weight by gradient descent algorithm.>>Repeat the training phase with updated weights.>>Make predictions.
