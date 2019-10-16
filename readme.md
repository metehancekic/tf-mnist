## Tensorflow Low-API Tutorial

**Why Low API ?**

- Because it allows you to modify standard structures as you wish easily 
In high API modification can be tedious depending on modification).

- Design your model more hands on..
Analyzing your model is easier.

**Tensorflow Elements**

There are 3 important things in tensorflow: 

1. placeholders : where you can feed data to the graph you build
2. variables    : trainable parameters of graph ( weights, biases, etc)
3. hyper-parameters : parameters that we tune manually ( learning rate, number of iterations, regularization term, keeping prob, etc. )

**How Tensorflow Works ?**

1. Build your graph.

   - Start with placeholders, you will use them to feed datasets (training dataset, validation etc.) or some parameters that you want to feed while executing your graph (training, testing etc.). 

   - Define the operations on top of your placeholders (convolutions, fully connected layers, max pooling, etc..).

   - Define your loss (Cross entropy, L2 loss, ...).

   - Define your optimizer (SGD, Adam, ...) and specify what you will minimize with it.

   - Define useful metrics for evaluation of the neural network (Prediction, Accuracy)

   - You are done with graph.. Congratulations...

2. Train your model.

   - Start a session (It can be interactive).

   - Initialize all the trainable parameters in your graph (sess.run(tf.global_variables_initializer()))

   - Run the optimizer in a loop with the session by feeding training dataset (Executes optimizer part of the graph which updates all the trainable parameters with backprop).

   - If you want to evaluate your model on the run, just run accuracy with session by feeding validation dataset.

   - **Important Note:** To be able to execute any node on your graph (optimizer, accuracy, etc...) you need to feed something to the placeholders that you defined in your graph. Graph is your system, placeholders are input to that system and all the nodes can be output to your system ( f(x) = y, x: placeholder, f(): graph, y: output ).

3. Evaluate your model.

   - Test your code by feeding test data.

   



