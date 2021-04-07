import tensorflow as tf

# Training Parameters
lr = 0.0005
steps = 50000
batch = 8

def normalise_data(s,d):

    return s,d

def denormalise (d):

    return d

def autoencoder1(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 8)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 8)
    encode = tf.nn.leaky_relu(encode, 0.2)
    # Encoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(encode, 4)
    
    return recon

def autoencoder2(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 8)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 16)
    encode = tf.nn.leaky_relu(encode, 0.2)
    # Encoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(encode, 4)
    
    return recon

def autoencoder3(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 16)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 16)
    encode = tf.nn.leaky_relu(encode, 0.2)
    # Encoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(encode, 4)
    
    return recon

def autoencoder4(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 8)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 32)
    encode = tf.nn.leaky_relu(encode, 0.2)
    
    # Encoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(encode, 4)


    return recon

def autoencoder5(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 16)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 32)
    encode = tf.nn.leaky_relu(encode, 0.2)
    
    # Encoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(encode, 4)
    
    return recon

def autoencoder6(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 32)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 32)
    encode = tf.nn.leaky_relu(encode, 0.2)
    # Encoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(encode, 4)

    
    return recon

def autoencoder7(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 64)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 64)
    encode = tf.nn.leaky_relu(encode, 0.2)
    # Encoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(encode, 4)
    
    return recon

def autoencoder8(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 128)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 128)
    encode = tf.nn.leaky_relu(encode, 0.2)
    
    # Encoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(encode, 4)
    
    return recon

def autoencoder9(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 128)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 256)
    encode = tf.nn.leaky_relu(encode, 0.2)
    
    # Encoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(encode, 4)
    
    return recon
