import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class Reduce_DataSet(object):
    '''This class contains all the methods needed to convert the base 10
        MNIST dataset into a base n dataset
        entire_mnist - user loads from Notebook: input_data.read_sets(path, one_hot= True)
        allowed_classes - list of '''

    def __init__(self, entire_mnist, allowed_classes):
        self.entire_mnist = entire_mnist
        self.train_images = entire_mnist.train.images
        self.train_labels = entire_mnist.train.labels
        self.test_images = entire_mnist.test.images
        self.test_labels = entire_mnist.test.labels
        self.allowed_classes = allowed_classes

    def _get_indices(self, train= True):
        '''Can retrieve indices of desired mnist values (e.g. indexes where label
        is either 2 or 9) for either Train or Test set.
        Default is to retrieve train data indices
        '''
        if train:
            pos_indicator = np.where(self.train_labels[:] == 1)[1]
        else:
            pos_indicator = np.where(self.test_labels[:] == 1)[1]
        t_indices = np.where(reduce(np.logical_or, [x== pos_indicator for x in self.allowed_classes]))
        return t_indices

    def reduce_train_test_set(self, train= True):
        '''Using indices found above, parse out images and labels for reduced set
        '''
        if train:
            t_indices = self._get_indices()
            X = self.train_images[t_indices]
            y = self.train_labels[t_indices]
        else:
            t_indices = self._get_indices(False)
            X = self.test_images[t_indices]
            y = self.test_labels[t_indices]
        return X, y

    def fix_y_encoding(self, train= True):
        '''
        Changes old one-hot encoding from 10-class system to n-class system
        e.g. [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] -> [1, 0]
        and  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] -> [0, 1]
        converts 2 and 9 encoded vectors from base 10 to base 2
        y_old_encode: m x 10 np.array of m data encoded in base 10
        labels: python list of labels in desired classifier'''
        if train:
            y_old_encode = self.reduce_train_test_set()[1]
        else:
            y_old_encode = self.reduce_train_test_set(False)[1]
        num_classes = len(self.allowed_classes)
        y = np.zeros((y_old_encode.shape[0], num_classes))
        pos_indicator = np.where(y_old_encode[:] == 1)[1]
        # t_indices = np.where(reduce(np.logical_or, [x== pos_indicator for x in lst]))
        for i, x in enumerate(self.allowed_classes):
            new_enc = np.zeros(num_classes)
            new_enc[i] = 1
            y[pos_indicator == x] = new_enc
            #y[t_indices] = new_enc
        return y

class Plot(object):
    '''
    labels_old_encode = Reduce_DataSet(entire_mnist, allowed_classes).reduce_train_test_set()[1]
        - these can be train or test labels, but need old encoding so that argmax
         corresponds to same handwritten number that image is labeled as
    '''
    def __init__(self, my_images, labels_old_encode):
        self.my_images = my_images
        self.my_labels = np.argmax(labels_old_encode, axis= 1)

    def _remove_grid_lines(self, axes):
        """Remove the default grid lines from a collection of axies."""
        for ax in axes.flatten():
            ax.grid(False)

    def _plot_greyscale_image(self, image, label, ax):
        """Plot a greyscale image and label its class."""
        first_digit = image.reshape(28, -1)
        ax.imshow(first_digit, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Class: {}".format(label))

    def plot_images(self, axes):
        for image, label, ax in zip(self.my_images, self.my_labels,
                                    axes.flatten()):
            self._plot_greyscale_image(image, label, ax)

        self._remove_grid_lines(axes)

class Evaluation(object):

    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
