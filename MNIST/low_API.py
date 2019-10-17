'''
Author : Metehan Cekic
Date : 04/14/2019
'''

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data      # To read MNIST data 
from tensorflow.python.tools import inspect_checkpoint as chkp  # To be able to check what do we have in checkpoints

import numpy as np
from tqdm import tqdm     # Adds Progress bar to loops


'''
Why Low API?

Because it allows you to modify standard structures as you wish easily 
In high API modification can be tedious depending on modification).

Design your model more hands on..
Analyzing your model is easier.

'''


'''
There are 3 important things in tensorflow: 

1) placeholders : where you can feed data to the graph you build
2) variables    : trainable parameters of graph ( weights, biases, etc)
3) hyper-parameters : parameters that we tune manually ( learning rate, number of iterations, regularization term, keeping prob, etc. )

'''
        
def model(x, prob):                                         # Neural Model

    with tf.variable_scope("CNN", reuse = tf.AUTO_REUSE):       

    # Variable scopes are important, it allows tensorflow to distinguish the tensors with same name

        with tf.variable_scope("layer1", reuse = tf.AUTO_REUSE):

            w1 = tf.get_variable("weight",[5,5,1,64],regularizer = tf.contrib.layers.l2_regularizer(0.001))
            # w1 : is in variable scope "CNN/layer1/weight"
            # [x axis, y axis, input channel, output channel]
            b1 = tf.get_variable("bias",  64 )
            # [output channel]
            o1 = tf.nn.relu(tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME') - b1)
            # [batch, x axis, y axis, channel]
            o1 = tf.nn.max_pool(value = o1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME" )
        
        with tf.variable_scope("layer2", reuse = tf.AUTO_REUSE):
            w2 = tf.get_variable("weight",[5,5,64,128],regularizer = tf.contrib.layers.l2_regularizer(0.001))
            b2 = tf.get_variable("bias",  128 )

            o2 = tf.nn.relu(tf.nn.conv2d(o1,w2,strides=[1,1,1,1],padding='VALID') - b2)
            o2 = tf.nn.max_pool(value = o2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME" )
            o2 = tf.contrib.layers.flatten(o2)
            
        with tf.variable_scope("layer3", reuse = tf.AUTO_REUSE):
            w3 = tf.get_variable("weight",[3200, 300] ,regularizer = tf.contrib.layers.l2_regularizer(0.001))
            b3 = tf.get_variable("bias",  300 )
            o3 = tf.nn.relu(tf.matmul(o2,w3) - b3)
            o3 = tf.nn.dropout(o3, keep_prob = prob)

            
        with tf.variable_scope("layer4", reuse = tf.AUTO_REUSE):
            w4 = tf.get_variable("weight",[300, 100] ,regularizer = tf.contrib.layers.l2_regularizer(0.001))
            b4 = tf.get_variable("bias",  100 )
            o4 = tf.nn.relu(tf.matmul(o3,w4) - b4)
            
        with tf.variable_scope("output", reuse = tf.AUTO_REUSE):
            w5 = tf.get_variable("weight",[100, 10] ,regularizer = tf.contrib.layers.l2_regularizer(0.001))
            b5 = tf.get_variable("bias",  10 )
            o5 = tf.matmul(o4,w5) - b5
        
    return o5 #,w1  # To be able to inspect w1, you need to give as output of this function, and run a session to get values.


def main():                                         # Main function to run
    
    tf.reset_default_graph()                        # To reset any existing graphs
    tf.logging.set_verbosity(tf.logging.ERROR)      # If you don't want to see too much warnings, put this !!

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # print(mnist.test.images.shape)
    # print(mnist.train.images.shape)

    images, labels = mnist.train.next_batch(100)

    # print(images.shape)
    # print(labels.shape)
    # print(mnist.validation.images.shape)          # Checks the shape of datasets

    # plt.imshow(images[0].reshape([28,28]))        # Plot a random image from dataset
    # plt.show()

    ################# BUILDING GRAPH ###################
    # Placeholders are input nodes for our graph.

    x = tf.placeholder(tf.float32, shape = [None, 28, 28, 1], name ="inputs")
    y_actual = tf.placeholder(tf.float32, shape = [None, 10], name ="labels")
    prob = tf.placeholder_with_default(1.0, shape=(), name = "prob")  # Placeholder with default value, 
    # if you don't specify in feed_dict it will assume prob = 1.0


    # We need logits to get predictions.
    logits = model(x, prob)
    # logits, w1 = model(x, prob)

    Y = tf.nn.softmax(logits)      

    # Cross Entropy takes logits and actual labels as input.
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels = y_actual, logits = logits)
    total_loss = tf.losses.get_total_loss(add_regularization_losses = True)


    ## OPTIMIZER: Updates weights according to backprop algorithm
    # There are options: ADAM, SGD, MOMENTUM SGD ...

    # optimizer = tf.train.AdamOptimizer( learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-3).minimize(total_loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(total_loss)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum = 0.1).minimize(total_loss)

    ## We need to assess our neural net, whether it learns or not. We can evaluate our model by defining accuracy node which uses prediction node.. 

    prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_actual, 1)) # Don't forget we have one hot labels !!
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))  # Just get the accuracy

    # GRAPH IS DONE.. What else?
    # Let's execute the graph by feeding data.

    # We need to start a session
    sess = tf.InteractiveSession()

    # Initialize all the variables in our graph.
    sess.run(tf.global_variables_initializer())

    nb_classes = 10 # Number of classes

    # Hyper parameters for our model
    num_iter = 200
    load_from_checkpoint = False     # If you don't want to train and just use a checkpoint, set this to true
    batch_size = 100

    if not load_from_checkpoint:
        for i in tqdm(range(num_iter)):     # tqdm shows a progress bar on your terminal for the for loop..

            images, labels = mnist.train.next_batch(batch_size) # Get batches to feed in and execute "optimizer" node

            train_data = {                                      # We need to feed data as a dictionary ( placeholders will be keys of   
                x: images.reshape([-1,28,28,1]),                # dictionary ) to our graph to execute a specific node.
                y_actual: labels,
                prob: 0.5}                                      # Keep probability of dropout

            sess.run(optimizer, feed_dict=train_data)           # Optimizer is executed here


            if i % np.int(num_iter/10) == 0:                    # Every num_iter/10 th iteration we check validation accuracy

                validation_data = {
        		    x: mnist.validation.images.reshape([-1,28,28,1]), 
        		    y_actual: mnist.validation.labels}

                acc_val = sess.run(accuracy, feed_dict=validation_data)

                print('\nValidation accuracy: {0:.2f}%'.format(acc_val*100))
            

        test_data = {
            x: mnist.test.images.reshape([-1,28,28,1]), 
            y_actual: mnist.test.labels}

        acc = sess.run(accuracy, feed_dict=test_data)       # After training we evaluate model by computing test accuracy

        print('Final Test accuracy: {0:.2f}% '.format(acc*100))

        save_dir = "./checkpoints/mnist" + ".ckpt"          # Saving checkpoints in following folder
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_dir)              # Save checkpoints

    else:
     
        res_dir = "./checkpoints/mnist" + ".ckpt"
        saver = tf.train.Saver(var_list = tf.trainable_variables("CNN"))   # Restoring checkpoints in following folder

        # chkp.print_tensors_in_checkpoint_file(res_dir, tensor_name='', all_tensors=True)                      # Inspect checkpoint (Prints out all the tensors saved in checkpoint)
        # chkp.print_tensors_in_checkpoint_file(res_dir, tensor_name='CNN/layer1/weight', all_tensors=False)    # Inspect checkpoint (Prints out tensors with specified name saved in checkpoint)

        saver.restore(sess, res_dir)                        # Restore checkpoints

        test_data = {
            x: mnist.test.images.reshape([-1,28,28,1]), 
            y_actual: mnist.test.labels}

        acc = sess.run(accuracy, feed_dict=test_data)

        print('Checkpoint Test accuracy: {0:.2f}% '.format(acc*100))


    # w1 = sess.run(w1, feed_dict = test_data)          # If you want to inspect any parameter defined in graph you can just run the 
                                                        # graph for that variable

if __name__ == '__main__':                          # When you run low_API.py, following code will be executed
    
    main()





