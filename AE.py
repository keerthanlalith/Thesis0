import tensorflow as tf

# Training Parameters
lr = 0.00005
steps = 50000
batch = 64

def normalise_data(s,ns):

    return s,ns

def denormalise (ns):

    return ns

def autoencoder1(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 8)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 8)
    encode = tf.nn.leaky_relu(encode, 0.2)
    # Encoder Hidden layer with leaky_relu activation #3
    latent = tf.layers.dense(encode, 8)
        
    # Decoder Hidden layer with leaky_relu activation #1
    decode = tf.layers.dense(latent, 8)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #2
    decode = tf.layers.dense(decode, 8)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(decode, 4)
    
    return recon

def autoencoder2(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 8)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 16)
    encode = tf.nn.leaky_relu(encode, 0.2)
    # Encoder Hidden layer with leaky_relu activation #3
    latent = tf.layers.dense(encode, 8)
        
    # Decoder Hidden layer with leaky_relu activation #1
    decode = tf.layers.dense(latent, 16)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #2
    decode = tf.layers.dense(decode, 8)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(decode, 4)
    
    return recon

def autoencoder3(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 16)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 16)
    encode = tf.nn.leaky_relu(encode, 0.2)
    # Encoder Hidden layer with leaky_relu activation #3
    latent = tf.layers.dense(encode, 8)
        
    # Decoder Hidden layer with leaky_relu activation #1
    decode = tf.layers.dense(latent, 16)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #2
    decode = tf.layers.dense(decode, 16)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(decode, 4)
    
    return recon

def autoencoder4(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 8)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 32)
    encode = tf.nn.leaky_relu(encode, 0.2)
    
    # Encoder Hidden layer with leaky_relu activation #3
    latent = tf.layers.dense(encode, 8)
    
    # Decoder Hidden layer with leaky_relu activation #1
    decode = tf.layers.dense(latent, 32)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #2
    decode = tf.layers.dense(decode, 8)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(decode, 4)

    return recon

def autoencoder5(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 16)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 32)
    encode = tf.nn.leaky_relu(encode, 0.2)
    
    # Encoder Hidden layer with leaky_relu activation #3
    latent = tf.layers.dense(encode, 8)
    
    # Decoder Hidden layer with leaky_relu activation #1
    decode = tf.layers.dense(latent, 32)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #2
    decode = tf.layers.dense(decode, 16)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(decode, 4)
    
    return recon

def autoencoder6(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 32)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 32)
    encode = tf.nn.leaky_relu(encode, 0.2)
    # Encoder Hidden layer with leaky_relu activation #3
    latent = tf.layers.dense(encode, 8)
        
    # Decoder Hidden layer with leaky_relu activation #1
    decode = tf.layers.dense(latent, 32)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #2
    decode = tf.layers.dense(decode, 32)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(decode, 4)
    
    return recon

def autoencoder7(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 64)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 64)
    encode = tf.nn.leaky_relu(encode, 0.2)
    # Encoder Hidden layer with leaky_relu activation #3
    latent = tf.layers.dense(encode, 8)
        
    # Decoder Hidden layer with leaky_relu activation #1
    decode = tf.layers.dense(latent, 64)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #2
    decode = tf.layers.dense(decode, 64)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #3
    recon = tf.layers.dense(decode, 4)
    
    return recon

def autoencoder8(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 128)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 128)
    encode = tf.nn.leaky_relu(encode, 0.2)
    
    # Encoder Hidden layer with leaky_relu activation #3
    latent = tf.layers.dense(encode, 8)
        
    # Decoder Hidden layer with leaky_relu activation #1
    decode = tf.layers.dense(latent, 128)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #2
    decode = tf.layers.dense(decode, 128)
    decode = tf.nn.leaky_relu(decode, 0.2)

    recon = tf.layers.dense(decode, 4)
    
    return recon

def autoencoder9(input_l):
    # Encoder Hidden layer with leaky_relu activation #1
    encode = tf.layers.dense(input_l, 128)
    encode = tf.nn.leaky_relu(encode, 0.2)

    # Encoder Hidden layer with leaky_relu activation #2
    encode = tf.layers.dense(encode, 256)
    encode = tf.nn.leaky_relu(encode, 0.2)
    
    # Encoder Hidden layer with leaky_relu activation #3
    latent = tf.layers.dense(encode, 8)
        
    # Decoder Hidden layer with leaky_relu activation #1
    decode = tf.layers.dense(latent, 256)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer with leaky_relu activation #2
    decode = tf.layers.dense(decode, 128)
    decode = tf.nn.leaky_relu(decode, 0.2)

    # Decoder Hidden layer
    recon = tf.layers.dense(decode, 4)
    
    return recon