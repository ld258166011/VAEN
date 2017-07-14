# ===================================================
# Figure drawing helper functions called by process
# in 'main.py' using Matplotlib.
#
# Copyright 2017 by Ding Li. All Rights Reserved.
# ===================================================

import numpy
import matplotlib.pyplot as plt

def draw_x_reconstruct(model, datasets, filename, load_params=True, save_data=False):
    VAEN = model
    IMTD = datasets
    shapes = datasets.test.shapes
    # Load parameters
    if load_params:
        VAEN.LoadALL("params/s1param")    
    # Reconstruct
    x_samples, _ = IMTD.s2train.next_batch(100, shuffle=False, rewind=True)
    epsilon = numpy.random.normal(loc=0, scale=1, size=[100, VAEN.network['n_latent']])
    x_reconstructions = VAEN.reconstruct(x_samples, epsilon)
    # Draw chosen images
    print("Drawing", filename)
    chosen_index = (0, 1, 9)
    draw_num = 0
    plt.figure(figsize=(8, 8))
    for i in chosen_index:
        x_original = x_samples[i].reshape(shapes[0], shapes[1])
        x_reconstruction = x_reconstructions[i].reshape(shapes[0], shapes[1])
        # Save reconstruction image
        if save_data:
            numpy.save("%s_n%d" % (filename, i), x_reconstruction)
        # Draw images
        plt.subplot(3, 2, 2 * draw_num + 1)
        plt.imshow(x_original, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        if draw_num == 0:
            plt.title("Original")
        plt.subplot(3, 2, 2 * draw_num + 2)
        plt.imshow(x_reconstruction, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        if draw_num == 0:
            plt.title("Reconstruction")
        draw_num += 1
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)
    plt.close()

def draw_x_recognize(model, datasets, filename, load_params=True, save_data=False):
    VAEN = model
    IMTD = datasets
    # Draw only when Dim(z) = 2
    if VAEN.network["n_latent"] != 2:
        return
    # Load parameters
    if load_params:
        VAEN.LoadALL("params/s1param")
    # Recognize
    x_samples, x_labels = IMTD.test.next_batch(10000, shuffle=False, rewind=True)
    z_means = VAEN.recognize(x_samples)
    # Save recognize result
    if save_data:
        numpy.save(filename, z_means)
    # Draw figure
    print("Drawing", filename)
    class_num = 12
    class_marker = ['^', 'v', '<', '>', 's', 'D', 'p', 'h', '8', 'P', 'X', '*', 'o']
    color = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple',
             'olive', 'brown', 'lime', 'fuchsia', 'mediumspringgreen', 'whitesmoke']
    plt.figure(figsize=(4, 3.2))
    bg = x_labels.sum(1) == 0
    count = len(bg[bg])
    size = numpy.ones(count) * 15
    marker = class_marker[class_num]
    plt.scatter(z_means[bg, 0], z_means[bg, 1],
                s=size, c=color[class_num],
                alpha=0.3, marker=marker,
                lw=0.5, edgecolors='black')         
    dots = list()
    labels = list()
    for i in range(class_num):
        app = x_labels[:, i] == 1
        count = len(app[app])
        size = numpy.ones(count) * 60         
        marker = class_marker[i]
        plt.scatter(z_means[app, 0], z_means[app, 1],
                    s=size, c=color[i],
                    alpha=0.5, marker=marker,
                    lw=0.5, edgecolors='black')
        dot, = plt.plot(4, 4, lw=0, marker=class_marker[i],
                        markerfacecolor=color[i], markeredgecolor='black',
                        markeredgewidth=0.5, markersize=7.5)
        dots.append(dot)
        labels.append(str(i))
    dot, = plt.plot(4, 4, lw=0, marker=class_marker[class_num],
                    markerfacecolor=color[class_num], markeredgecolor='black',
                    markeredgewidth=0.5, markersize=4)
    dots.append(dot)
    labels.append(str(class_num))
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xlabel("z (axis 0)")
    plt.ylabel("z (axis 1)")    
    plt.grid(linestyle=":", linewidth=0.5)       
    plt.subplots_adjust(left=0.15, right=0.8, bottom=0.15, top=0.95, wspace=0, hspace=0)
    plt.legend(dots, labels,
               title='Labels', markerfirst=False,
               loc=1, bbox_to_anchor=(1.26, 1),
               labelspacing=0.31, borderaxespad=0., handletextpad=0.1)
    plt.savefig(filename, dpi=1000)
    plt.close()

def draw_x_generate(model, datasets, filename, load_params=True):
    VAEN = model
    shapes = datasets.test.shapes
    # Draw only when Dim(z) = 2
    if VAEN.network["n_latent"] != 2:
        return
    # Load parameters
    if load_params:
        VAEN.LoadALL("params/s1param")
    # Draw images
    print("Drawing", filename)
    n_z0 = 7
    n_z1 = 7
    z0s = numpy.linspace(-3, 3, n_z0)
    z1s = numpy.linspace(-3, 3, n_z1)
    canvas = numpy.ones(((shapes[0]+1)*n_z1-1, (shapes[1]+1)*n_z0-1))
    for z0, z0i in enumerate(z0s):
        for z1, z1i in enumerate(z1s):
            z_samples = numpy.array([[z1i, z0i]])
            # Generate
            x_reconstruction = VAEN.generate(z_samples)
            canvas[(n_z0-z0-1)*(shapes[0]+1):(n_z0-z0)*(shapes[0]+1)-1,
                   z1*(shapes[1]+1):(z1+1)*(shapes[1]+1)-1] = \
            x_reconstruction[0].reshape(shapes[0], shapes[1])
    plt.figure(figsize=(4, 3.2))
    plt.imshow(canvas, cmap="gray")
    xticks = numpy.arange(7)*(shapes[1]+1)+shapes[1]/2
    yticks = numpy.arange(7)*(shapes[0]+1)+shapes[0]/2
    labels = list()
    for i in range(-3, 4):
        labels.append(str(i))
    plt.xticks(xticks, labels)
    plt.yticks(yticks, labels)
    plt.xlabel("z (axis 0)")
    plt.ylabel("z (axis 1)")
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.97, wspace=0, hspace=0)
    plt.savefig(filename, dpi=7000)
    plt.close()

def draw_costs(costs, function, filename, save_data=False):
    # Save costs
    if save_data:
        numpy.save(filename, costs)
    # Draw figure
    print("Drawing", filename)
    epochs = numpy.array(range(len(costs))) + 1
    plt.figure(figsize=(4, 3))
    if function == "plot":
        plt.plot(epochs, costs, linewidth=1)
    elif function == "semilogy":
        plt.semilogy(epochs, costs, basey = 2, linewidth=1)
    plt.xlabel("Training Epoch")
    plt.ylabel("Model Cost")
    plt.grid(axis="y", linestyle=":", linewidth=1)
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)
    plt.close()
    
def draw_evaluations(base, thresholds, accuracys, precisions, recalls, filename, save_data=True):
    # Save costs
    if save_data:
        numpy.save(filename + "_t", thresholds)
        numpy.save(filename + "_a", accuracys)
        numpy.save(filename + "_p", precisions)
        numpy.save(filename + "_r", recalls)
    # Draw figure
    print("Drawing", filename)
    plt.figure(figsize=(4, 3))
    start = 21
    last = 13
    plt.plot(precisions[start:-last], linewidth=1, label="Precision")
    plt.plot(recalls[start:-last], linewidth=1, label="Recall")
    plt.plot(accuracys[start:-last], linewidth=1, label="Accuracy")
    plt.xlabel("Threshold")
    plt.xticks()
    plt.grid(axis="y", linestyle=":", linewidth=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)
    plt.close()
    