# ==========================================================
# Variational Autoencoder Network model with a sklearn-like
# interface implemented using TensorFlow.
#
# Copyright 2017 by Ding Li. All Rights Reserved.
# ==========================================================

import numpy
import tensorflow as tf

class VariationalAutoencoderNetwork(object):

    def __init__(self, network_architectrue, activation_fun=tf.nn.softplus, learning_rate=0.001):
        self.network = network_architectrue
        self.activation_fun = activation_fun
        self.learning_rate = learning_rate
        
        self.x = tf.placeholder(tf.float32, [None, self.network["n_input"]])
        self.epsilon = tf.placeholder(tf.float32, [None, self.network["n_latent"]])
        self.y = tf.placeholder(tf.float32, [None, self.network["n_class"]])
        
        self._create_network()
        self._create_optimizer()
        
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    
    def _create_network(self):
        network_params = self._initialize_params()
        
        # Encode input data into latent space value
        self.z = self._recognition_network(network_params)
        
        # Decode latent space value into reconstruction data
        self.x_reconstruct = self._generation_network(network_params)
        
        # Predict class of input data
        self.y_pridict = self._classifier(network_params)        
    
    def _initialize_params(self):
        def xavier_variable(name, shape):
            return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
        
        n_mlp_layer = self.network["n_mlp_layer"]
        if n_mlp_layer < 1 and n_mlp_layer > 10:
            raise ValueError("Error! MLP Layer out of range")
        
        n_input = self.network["n_input"]
        n_mlp_neuron = self.network["n_mlp_neuron"]
        n_latent = self.network["n_latent"]
        n_class = self.network["n_class"]          
        
        params = dict()
        # Recognition
        params["recog_W1"] = xavier_variable("recog_W1", [n_input, n_mlp_neuron])
        for i in range(2, n_mlp_layer + 1):
            params["recog_W%d" % i] = xavier_variable("recog_W%d" % i, [n_mlp_neuron, n_mlp_neuron])
        for i in range(1, n_mlp_layer + 1):
            params['recog_b%d' % i] = tf.Variable(tf.zeros([n_mlp_neuron], dtype=tf.float32))
        params["recog_Wz_mean"] = xavier_variable("recog_Wz_mean", [n_mlp_neuron, n_latent])
        params["recog_bz_mean"] = tf.Variable(tf.zeros([n_latent], dtype=tf.float32))
        params["recog_Wz_sigma"] = xavier_variable("recog_Wz_sigma", [n_mlp_neuron, n_latent])
        params["recog_bz_sigma"] = tf.Variable(tf.zeros([n_latent], dtype=tf.float32))
        
        # Generation
        params["gener_W1"] = xavier_variable("gener_W1", [n_latent, n_mlp_neuron])
        for i in range(2, n_mlp_layer + 1):
            params["gener_W%d" % i] = xavier_variable("gener_W%d" % i, [n_mlp_neuron, n_mlp_neuron])
        for i in range(1, n_mlp_layer + 1):
            params['gener_b%d' % i] = tf.Variable(tf.zeros([n_mlp_neuron], dtype=tf.float32))
        params["gener_Wx"] = xavier_variable("gener_Wx", [n_mlp_neuron, n_input])
        params["gener_bx"] = tf.Variable(tf.zeros([n_input], dtype=tf.float32))
        
        # Classification
        params["class_W"] = tf.Variable(tf.zeros([n_latent, n_class], dtype=tf.float32))
        params["class_b"] = tf.Variable(tf.zeros([n_class], dtype=tf.float32))
        
        return params
    
    def _recognition_network(self, params):
        mlp = dict()
        n_L = self.network["n_mlp_layer"]
        mlp["L1"] = self.activation_fun(tf.add(tf.matmul(self.x,params["recog_W1"]),
                                               params["recog_b1"]))
        for i in range(2, n_L + 1):
            mlp["L%d" % i] = self.activation_fun(tf.add(tf.matmul(mlp["L%d" % (i - 1)], params["recog_W%d" % i]),
                                                        params["recog_b%d" % i]))
        self.z_mean = tf.add(tf.matmul(mlp["L%d" % n_L], params['recog_Wz_mean']),
                             params['recog_bz_mean'])
        self.z_log_sigma_sq = tf.add(tf.matmul(mlp["L%d" % n_L], params['recog_Wz_sigma']),
                                     params['recog_bz_sigma'])
        z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), self.epsilon))
        return z
    
    def _generation_network(self, params):
        mlp = dict()
        n_L = self.network["n_mlp_layer"]
        mlp["L1"] = self.activation_fun(tf.add(tf.matmul(self.z, params["gener_W1"]),
                                               params["gener_b1"]))
        for i in range(2, n_L + 1):
            mlp["L%d" % i] = self.activation_fun(tf.add(tf.matmul(mlp["L%d" % (i - 1)], params["gener_W%d" % i]),
                                                        params["gener_b%d" % i]))
        x_reconstruct = tf.nn.sigmoid(tf.add(tf.matmul(mlp["L%d" % n_L], params['gener_Wx']),
                                             params['gener_bx']))
        return x_reconstruct
    
    def _classifier(self, params):
        y_pridict = tf.nn.softmax(tf.add(tf.matmul(self.z_mean, params['class_W']), params['class_b']))
        return y_pridict
    
    def _create_optimizer(self):
        # Stage 1
        square_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_reconstruct, self.x), 2.0), 1)
        entropy_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstruct)
                                      + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstruct), 1)
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.losses = {"s_loss" : tf.reduce_mean(square_loss),
                       "e_loss" : tf.reduce_mean(entropy_loss),
                       "l_loss" : tf.reduce_mean(latent_loss)}
        self.s1cost = tf.reduce_mean(entropy_loss + latent_loss)
        self.s1optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.s1cost)
        
        # Stage 2
        entropy_loss = -tf.reduce_sum(self.y * tf.log(1e-10 + self.y_pridict)
                                      + (1-self.y) * tf.log(1e-10 + 1 - self.y_pridict))
        self.s2cost = entropy_loss
        self.s2optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.s2cost)
    
    def stage1_training(self, x, e):
        _, cost, losses = self.sess.run((self.s1optimizer, self.s1cost, self.losses),
                                        feed_dict={self.x: x, self.epsilon: e})
        return cost, losses
    
    def recognize(self, x):
        z_mean = self.sess.run(self.z_mean,
                               feed_dict={self.x: x})
        return z_mean
    
    def generate(self, z):
        x_reconstruct = self.sess.run(self.x_reconstruct,
                                      feed_dict={self.z: z})
        return x_reconstruct
    
    def reconstruct(self, x, e):
        x_reconstruct = self.sess.run(self.x_reconstruct,
                                      feed_dict={self.x: x, self.epsilon: e})
        return x_reconstruct
    
    def stage2_training(self, x, e, y):
        _, cost = self.sess.run((self.s2optimizer, self.s2cost),
                                feed_dict={self.x: x, self.epsilon: e, self.y: y})
        return cost
    
    def pridict(self, x):
        y_pridict = self.sess.run(self.y_pridict,
                                  feed_dict={self.x: x})
        return y_pridict
    
    def SaveALL(self, path):
        print("Saving", path)
        self.saver.save(self.sess, path)
    
    def LoadALL(self, path):
        print("Loading", path)
        self.saver.restore(self.sess, path)
        
