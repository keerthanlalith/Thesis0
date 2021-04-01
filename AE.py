import tensorflow as tf

# Training Parameters
lr = 0.00015
steps = 50000
batch = 64

def normalise_data(s,ns):

    return s,ns

def denormalise (ns):

    return ns

# Network Parameters
num_input = 4 # MNIST data input (img shape: 28*28)

num_hidden_1 = 16 # 1st layer num features
num_hidden_2 = 16 # 2nd layer num features
num_latent = 4 # latent layer num features (the latent dim)

def autoencoder1(input_l):
    # Encoder Hidden layer with sigmoid activation #1
    encode = tf.layers.dense(input_l, num_hidden_1,activation=tf.nn.tanh)

    # Encoder Hidden layer with sigmoid activation #2
    encode = tf.layers.dense(encode, num_hidden_2, activation=tf.nn.tanh)    
    
    # Encoder Hidden layer with sigmoid activation #3
    latent = tf.layers.dense(encode, num_latent, activation=tf.nn.tanh)
    
    # Decoder Hidden layer with sigmoid activation #1
    #decode = tf.layers.dense(latent, num_hidden_2 , activation=tf.nn.tanh)

    # Decoder Hidden layer with sigmoid activation #2
    #decode = tf.layers.dense(decode, num_hidden_1, activation=tf.nn.tanh)

    # Decoder Hidden layer with sigmoid activation #3
    #recon = tf.layers.dense(decode, num_input, activation=tf.nn.tanh)
    
    return latent

def autoencoder2(input_l):
    # Encoder Hidden layer with sigmoid activation #1
    encode = tf.layers.dense(input_l, num_hidden_1,activation=tf.nn.tanh)

    # Encoder Hidden layer with sigmoid activation #2
    encode = tf.layers.dense(encode, num_hidden_2, activation=tf.nn.tanh)    
    
    # Encoder Hidden layer with sigmoid activation #3
    latent = tf.layers.dense(encode, num_latent, activation=tf.nn.tanh)
    
    # Decoder Hidden layer with sigmoid activation #1
    #decode = tf.layers.dense(latent, num_hidden_2 , activation=tf.nn.tanh)

    # Decoder Hidden layer with sigmoid activation #2
    #decode = tf.layers.dense(decode, num_hidden_1, activation=tf.nn.tanh)

    # Decoder Hidden layer with sigmoid activation #3
    #recon = tf.layers.dense(decode, num_input)
    
    return latent

def autoencoder3(input_l):
    # Encoder Hidden layer with sigmoid activation #1
    encode = tf.layers.dense(input_l, num_hidden_1,activation=tf.nn.tanh)

    # Encoder Hidden layer with sigmoid activation #2
    encode = tf.layers.dense(encode, num_hidden_2, activation=tf.nn.tanh)    
    
    # Encoder Hidden layer with sigmoid activation #3
    latent = tf.layers.dense(encode, num_latent)
    
    # Decoder Hidden layer with sigmoid activation #1
    #decode = tf.layers.dense(latent, num_hidden_2 , activation=tf.nn.tanh)

    # Decoder Hidden layer with sigmoid activation #2
    #decode = tf.layers.dense(decode, num_hidden_1, activation=tf.nn.tanh)

    # Decoder Hidden layer with sigmoid activation #3
    #recon = tf.layers.dense(decode, num_input)
    
    return latent

def autoencoder4(input_l):
    # Encoder Hidden layer with sigmoid activation #1
    encode = tf.layers.dense(input_l, num_hidden_1)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with sigmoid activation #2
    encode = tf.layers.dense(encode, num_hidden_2)
    encode = tf.nn.leaky_relu(encode, 0.2)
    
    # Encoder Hidden layer with sigmoid activation #3
    latent = tf.layers.dense(encode, num_latent)
    latent = tf.nn.leaky_relu(latent, 0.2)

    
    # Decoder Hidden layer with sigmoid activation #1
    #decode = tf.layers.dense(latent, num_hidden_2)
    #decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with sigmoid activation #2
    #decode = tf.layers.dense(decode, num_hidden_1)
    #decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with sigmoid activation #3
    #recon = tf.layers.dense(decode, num_input)
    #recon = tf.nn.leaky_relu(recon, 0.2)

    return latent

def autoencoder5(input_l):
    # Encoder Hidden layer with sigmoid activation #1
    encode = tf.layers.dense(input_l, num_hidden_1)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with sigmoid activation #2
    encode = tf.layers.dense(encode, num_hidden_2)
    encode = tf.nn.leaky_relu(encode, 0.2)
    
    # Encoder Hidden layer with sigmoid activation #3
    latent = tf.layers.dense(encode, num_latent)
    latent = tf.nn.leaky_relu(latent, 0.2)

    
    # Decoder Hidden layer with sigmoid activation #1
    #decode = tf.layers.dense(latent, num_hidden_2)
    #decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with sigmoid activation #2
    #decode = tf.layers.dense(decode, num_hidden_1)
    #decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with sigmoid activation #3
    #recon = tf.layers.dense(decode, num_input)
    
    return latent

def autoencoder6(input_l):
    # Encoder Hidden layer with sigmoid activation #1
    encode = tf.layers.dense(input_l, num_hidden_1)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with sigmoid activation #2
    encode = tf.layers.dense(encode, num_hidden_2)
    encode = tf.nn.leaky_relu(encode, 0.2)
    encode = tf.layers.dense(encode, num_hidden_2)
    encode = tf.nn.leaky_relu(encode, 0.2)
    
    # Encoder Hidden layer with sigmoid activation #3
    latent = tf.layers.dense(encode, num_latent)
        
    # Decoder Hidden layer with sigmoid activation #1
    #decode = tf.layers.dense(latent, num_hidden_2)
    #decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with sigmoid activation #2
    #decode = tf.layers.dense(decode, num_hidden_1)
    #decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with sigmoid activation #3
    #recon = tf.layers.dense(decode, num_input)
    
    return latent

def autoencoder7(input_l):
    # Encoder Hidden layer with sigmoid activation #1
    encode = tf.layers.dense(input_l, num_hidden_1,activation=tf.nn.sigmoid)

    # Encoder Hidden layer with sigmoid activation #2
    encode = tf.layers.dense(encode, num_hidden_2, activation=tf.nn.sigmoid)    
    
    # Encoder Hidden layer with sigmoid activation #3
    latent = tf.layers.dense(encode, num_latent, activation=tf.nn.sigmoid)
    
    # Decoder Hidden layer with sigmoid activation #1
    #decode = tf.layers.dense(latent, num_hidden_2 , activation=tf.nn.sigmoid)

    # Decoder Hidden layer with sigmoid activation #2
    #decode = tf.layers.dense(decode, num_hidden_1, activation=tf.nn.sigmoid)

    # Decoder Hidden layer with sigmoid activation #3
    #recon = tf.layers.dense(decode, num_input, activation=tf.nn.sigmoid)
    
    return latent

def autoencoder8(input_l):
    # Encoder Hidden layer with sigmoid activation #1
    encode = tf.layers.dense(input_l, num_hidden_1,activation=tf.nn.sigmoid)

    # Encoder Hidden layer with sigmoid activation #2
    encode = tf.layers.dense(encode, num_hidden_2, activation=tf.nn.sigmoid)    
    
    # Encoder Hidden layer with sigmoid activation #3
    latent = tf.layers.dense(encode, num_latent, activation=tf.nn.sigmoid)
    
    # Decoder Hidden layer with sigmoid activation #1
    #decode = tf.layers.dense(latent, num_hidden_2 , activation=tf.nn.sigmoid)

    # Decoder Hidden layer with sigmoid activation #2
    #decode = tf.layers.dense(decode, num_hidden_1, activation=tf.nn.sigmoid)

    # Decoder Hidden layer with sigmoid activation #3
    #recon = tf.layers.dense(decode, num_input)
    
    return latent

def autoencoder9(input_l):
    # Encoder Hidden layer with sigmoid activation #1
    encode = tf.layers.dense(input_l, num_hidden_1,activation=tf.nn.sigmoid)

    # Encoder Hidden layer with sigmoid activation #2
    encode = tf.layers.dense(encode, num_hidden_2, activation=tf.nn.sigmoid)    
    
    # Encoder Hidden layer with sigmoid activation #3
    latent = tf.layers.dense(encode, num_latent)
    
    # Decoder Hidden layer with sigmoid activation #1
    #decode = tf.layers.dense(latent, num_hidden_2 , activation=tf.nn.sigmoid)

    # Decoder Hidden layer with sigmoid activation #2
    #decode = tf.layers.dense(decode, num_hidden_1, activation=tf.nn.sigmoid)

    # Decoder Hidden layer with sigmoid activation #3
    #recon = tf.layers.dense(decode, num_input)
    
    return latent