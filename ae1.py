from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import copy
from AE import autoencoder1 as autoencoder
from AE import *

# Training Parameters
learning_rate = lr
num_steps = steps
batch_size = batch

# Import data
try:
    # train set
    filename = 'Data/NState.npy'
    ns = pickle.load(open(filename, 'rb'))
    filename = 'Data/Action.npy'
    a = pickle.load(open(filename, 'rb'))
    filename = 'Data/State.npy'
    s = pickle.load(open(filename, 'rb'))
    filename = 'Data/Diff.npy'
    d = pickle.load(open(filename, 'rb'))

    # test set
    filename = 'Data/TNState.npy'
    test_ns = pickle.load(open(filename, 'rb'))
    filename = 'Data/TAction.npy'
    test_a = pickle.load(open(filename, 'rb'))
    filename = 'Data/TState.npy'
    test_s = pickle.load(open(filename, 'rb'))
    filename = 'Data/TDiff.npy'
    test_d = pickle.load(open(filename, 'rb'))

except:
    print("Load Data Failed")

# Normalised data and labels
print("Normalised Data")
s,ns = normalise_data(s,ns)



def next_batch(n, data, labels):

    idx=np.random.choice(len(data), size=n, replace=False)
    data_shuffle = data[idx] 
    labels_shuffle =labels[idx]

    return data_shuffle, labels_shuffle

display_step = 1000
examples_to_show = 10


# tf Graph input (only pictures)
X = tf.placeholder("float", [None, 4])
X_d = tf.placeholder("float", [None, 4])

# Construct model
decoder_op = autoencoder(X)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X_d

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, batch_ns = next_batch(batch_size,s,ns)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x,X_d: batch_ns})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    pause=0
    while pause==1:
        pause=1
    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    
    # Normalise the input 
    test_ms = copy.deepcopy(test_s)    
    test_mns = copy.deepcopy(test_ns)
    test_ms, test_mns = normalise_data(test_ms,test_mns)
    n=len(test_s)
    
    # Encode the current state and decode the next state
    pred_ns = sess.run(decoder_op, feed_dict={X: test_ms})
    pred_mns = copy.deepcopy(pred_ns)

    #print(type(test_mns),type(pred_ns))
    print ("AE diff output, Normalised Test diff,  Difference")
    d_err = np.abs(test_mns - pred_ns)
    
    for i in range(n):
        for j in range(4):
            pred_ns[i][j]=float("{:5.6f}".format(pred_ns[i][j]))
            test_mns[i][j]=float("{:5.6f}".format(test_mns[i][j]))
            d_err[i][j]=float("{:5.6f}".format(d_err[i][j]))

    for i in range(n):
        print(pred_ns[i],test_mns[i],d_err[i])
    print("------")
    
    # Denormalise the data
    pred_mns = denormalise (pred_mns)
    d_err_1 = np.abs(test_ns - pred_mns)

    for i in range(n):
        for j in range(4):
            pred_mns[i][j]=float("{:5.6f}".format(pred_mns[i][j]))
            test_ns[i][j]=float("{:5.6f}".format(test_ns[i][j]))
            d_err_1[i][j]=float("{:5.6f}".format(d_err_1[i][j]))

    for i in range(n):
        print(pred_mns[i],test_ns[i],d_err_1[i])
    print("------")

    p_err =np.mean(d_err*d_err,axis=0)
    p_err2 =np.mean(d_err_1*d_err_1,axis=0)
    
    print(p_err,p_err2)
    
