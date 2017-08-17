# ===================================================
# Variation Autoencoder Network implements on IMTD17
# datasets using TensorFlow.
#
# Copyright 2017 by Ding Li. All Rights Reserved.
# ===================================================

import numpy
# User modules
import imtd
import model
import figure

# Datasets
IMTD = imtd.read_data_sets('IMTD17')

# Network architecture
input_num = IMTD.test.shapes[0] * IMTD.test.shapes[1]
class_num = IMTD.test.labels.shape[1]
mlp_layer = 2
mlp_neuron = 700
latent_num = 300
network_architectrue = dict(n_input=input_num,       # dimensionality of input data
                            n_mlp_layer=mlp_layer,   # number of mlp layers
                            n_mlp_neuron=mlp_neuron, # number of mlp neurons                  
                            n_latent=latent_num,     # dimensionality of latent space            
                            n_class=class_num)       # dimensionality of predict class

# Model
VAEN = model.VariationalAutoencoderNetwork(network_architectrue)

def Stage1Training(train_epochs=256,
                   batch_size=100,
                   display_step=1,
                   figure_base=2,
                   save_params=True):
    if train_epochs == 0:
        return
    n_samples = IMTD.s1train.num_examples
    train_costs = list()
    exponent = 0
    # Training epochs
    for epoch in range(train_epochs):
        avg_s_loss = 0.
        avg_e_loss = 0.
        avg_l_loss = 0.
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = IMTD.s1train.next_batch(batch_size)
            epsilon = numpy.random.normal(loc=0, scale=1, size=[batch_size, latent_num])
            # Training using unlabeled data
            cost, losses = VAEN.stage1_training(batch_xs, epsilon)
            # Compute average cost
            avg_cost += cost / n_samples * batch_size
            avg_s_loss += losses["s_loss"] / n_samples * batch_size
            avg_e_loss += losses["e_loss"] / n_samples * batch_size
            avg_l_loss += losses["l_loss"] / n_samples * batch_size
        # Save cost value
        train_costs.append(avg_cost)
        # Display cost
        if epoch % display_step == (display_step - 1):
            print("Stage1 epoch:", '%03d' % (epoch + 1),
                  "cost=", "{:.3f}".format(avg_cost),
                  "s_loss=", "{:.3f}".format(avg_s_loss),
                  "e_loss=", "{:.3f}".format(avg_e_loss),
                  "l_loss=", "{:.3f}".format(avg_l_loss))
        # Draw figure
        if figure_base and epoch + 1 == pow(figure_base, exponent):
            fig_fname = "figures/reconstruct_e%d" % (epoch + 1)
            figure.draw_x_reconstruct(VAEN, IMTD, fig_fname, load_params=False)
            fig_fname = "figures/recognize_e%d" % (epoch + 1)
            figure.draw_x_recognize(VAEN, IMTD, fig_fname, load_params=False)            
            exponent += 1
    # Draw costs figure
    fig_fname = "figures/s1costs_l%d_e%d_b%d" % (mlp_layer, train_epochs, batch_size)
    figure.draw_costs(train_costs, "plot", fig_fname)
    # Save parameters
    if save_params:
        VAEN.SaveALL("params/s1param")

def Stage2Training(train_epochs=8192,
                   batch_size=100,
                   display_step=100,
                   load_params=True,
                   save_params=True):
    if train_epochs == 0:
        return
    # Load parameters
    if load_params:
        VAEN.LoadALL("params/s1param")
    n_samples = IMTD.s2train.num_examples
    train_costs = list()
    # Training epochs
    for epoch in range(train_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = IMTD.s2train.next_batch(batch_size)
            epsilon = numpy.random.normal(loc=0, scale=1, size=[batch_size, latent_num])
            # Training using labeled data
            cost = VAEN.stage2_training(batch_xs, epsilon, batch_ys)
            # Compute average cost
            avg_cost += cost / n_samples * batch_size
        # Save cost value
        train_costs.append(avg_cost)
        # Display cost
        if epoch % display_step == (display_step - 1):
            print("Stage2 epoch:", '%03d' % (epoch + 1),
                  "cost=", "{:.6f}".format(avg_cost))
    # Draw costs figure
    fig_fname = "figures/s2costs_e%d_b%d" % (train_epochs, batch_size)
    figure.draw_costs(train_costs, "semilogy", fig_fname)    
    # Save parameters
    if save_params:
        VAEN.SaveALL("params/s2param")

def FindThreshold(find_steps=7,
                  start_step=13,
                  split_step=9,
                  exponent_base=2,
                  load_params=True):
    if find_steps == 0:
        return    
    # Load parameters
    if load_params:
        VAEN.LoadALL("params/s2param")
    test_num = IMTD.test.images.shape[0]
    test_xs, test_ys = IMTD.test.next_batch(test_num, shuffle=False, rewind=True)
    thresholds = list()
    accuracys = list()
    precisions = list()
    recalls = list()
    # Test thresholds
    for step in range(find_steps):
        for split in range(split_step):
            exponent = -step - 1 - start_step - split * (1 / split_step)
            threshold = 1 - pow(exponent_base, exponent)
            thresholds.append(threshold)
            # Test using label data
            pridict_ys = VAEN.pridict(test_xs)
            
            # Part2: Total evaluation
            positive = test_ys.sum(axis=1) > 0
            negative = ~ positive
            over_threshold = pridict_ys.max(axis=1) > threshold
            below_threshold = ~ over_threshold
            max_equal = test_ys.argmax(axis=1) == pridict_ys.argmax(axis=1)
            
            true_positive = (positive & over_threshold & max_equal).astype(float).sum()
            true_negative = (negative & below_threshold).astype(float).sum()
            false_positive = positive.astype(float).sum() - true_positive
            false_negative = negative.astype(float).sum() - true_negative   
            
            accuracy = (true_positive + true_negative) / test_num
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            accuracys.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
    # Draw evaluations figure
    fig_fname = "figures/evaluations"
    figure.draw_evaluations(exponent_base, thresholds, accuracys, precisions, recalls, fig_fname)

def Test(threshold=0.99999794,
         load_params=True):
    # Load parameters
    if load_params:
        VAEN.LoadALL("params/s2param")
    test_num = IMTD.test.images.shape[0]
    test_xs, test_ys = IMTD.test.next_batch(test_num, shuffle=False, rewind=True)
    pridict_ys = VAEN.pridict(test_xs)
    
    positive = test_ys.sum(axis=1) > 0
    negative = ~ positive
    over_threshold = pridict_ys.max(axis=1) > threshold
    below_threshold = ~ over_threshold
    max_equal = test_ys.argmax(axis=1) == pridict_ys.argmax(axis=1)
    
    # Part1: Class evaluation
    results = list()
    for c in range(class_num):
        _positive = test_ys[:, c] > 0
        _negative = ~ _positive
        _over_threshold = pridict_ys[:, c] > threshold
        _below_threshold = ~ _over_threshold
        _max_equal = max_equal & (test_ys.argmax(axis=1) == c)
        
        _true_positive = (_positive & _over_threshold & _max_equal).astype(float).sum()
        _true_negative = (_negative & _below_threshold).astype(float).sum()
        _false_positive = _positive.astype(float).sum() - _true_positive
        _false_negative = _negative.astype(float).sum() - _true_negative
        
        _accuracy = (_true_positive + _true_negative) / test_num
        _precision = _true_positive / (_true_positive + _false_positive)
        _recall = _true_positive / (_true_positive + _false_negative)
        print("Label:", '%02d' % c,
              " A=", "{:.4f}".format(_accuracy),
              " P=", "{:.4f}".format(_precision),
              " R=", "{:.4f}".format(_recall))
    
    # Part2: Total evaluation
    true_positive = (positive & over_threshold & max_equal).astype(float).sum()
    true_negative = (negative & below_threshold).astype(float).sum()
    false_positive = positive.astype(float).sum() - true_positive
    false_negative = negative.astype(float).sum() - true_negative   
    
    accuracy = (true_positive + true_negative) / test_num
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    print("Total:   ",
          " A=", "{:.4f}".format(accuracy),
          " P=", "{:.4f}".format(precision),
          " R=", "{:.4f}".format(recall))

# Stage 1 - Unsupervised Learning
Stage1Training(train_epochs=256,      # number of training epoch (0 for pass)
               batch_size=100,        # number of sample per training
               display_step=1,        # cycle of displaying result
               figure_base=100,       # exponential base of figures (0 for no)
               save_params=True)      # whether saving stage 1 parameters
# Draw Figures
figure.draw_x_reconstruct(VAEN, IMTD, "figures/reconstruct")
figure.draw_x_recognize(VAEN, IMTD, "figures/recognize")
figure.draw_x_generate(VAEN, IMTD, "figures/generate")

# Stage 2 - Supervised Learning
Stage2Training(train_epochs=512,      # number of training epoch (0 for pass)
               batch_size=100,        # number of sample per training
               display_step=100,      # cycle of displaying result
               load_params=True,      # whether loading stage 1 parameters
               save_params=True)      # whether saving stage 2 parameters

# Find best threshold
FindThreshold(find_steps=7,        # number of finding step (0 for pass)
              start_step=13,       # number of pass step
              split_step=9,        # number of split per step
              exponent_base=2,     # exponential base of step
              load_params=True)    # whether loading stage 2 parameters

# Classification Test
Test(threshold=0.99999794,  # threshold for classification
     load_params=True)      # whether loading stage 2 parameters
