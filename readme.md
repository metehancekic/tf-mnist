## Tensorflow Low-API Tutorial

A simple tutorial of MNIST classification with 4 layer convolutional neural network (CNN). Neural network is implemented with low-api tensorflow modules. 

**Why Low API?**

- Because it allows you to modify standard structures as you wish easily 
In high API modification can be tedious depending on modification).

- Design your model more hands on..
Analyzing your model is easier.

**Tensorflow Elements**

There are 3 important things in tensorflow: 

1. placeholders : where you can feed data to the graph you build
2. variables    : trainable parameters of graph ( weights, biases, etc)
3. hyper-parameters : parameters that we tune manually ( learning rate, number of iterations, regularization term, keeping prob, etc. )
