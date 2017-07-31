import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import matplotlib
matplotlib.style.use('ggplot')
from pandas.plotting import scatter_matrix
import numpy as np
from pandas.plotting import radviz
from sklearn.model_selection import train_test_split

class Manipulation(object):
    def __init__(self, data):
        self.data = data
        self.X = self.data.data
        self.y = self.data.target
        self.label_names = self.data.target_names
        self.feature_names = self.data.feature_names

    def make_label_name_dict(self):
        name_dict = {}
        for i, label_name in enumerate(self.label_names):
            name_dict[i] = label_name
        return name_dict

    def create_df(self, split= False):
        '''
        '''
        if split:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
            test_species_names = self.label_names[y_test]
            test_df = pd.DataFrame(X_test, columns= self.feature_names)
            test_df['species'] = y_test
            test_df['species name'] = test_species_names
            train_species_names = self.label_names[y_train]
            train_df = pd.DataFrame(X_train, columns= self.feature_names)
            train_df['species'] = y_train
            train_df['species name'] = train_species_names
            return train_df, test_df
        else:
            species_names = self.label_names[self.y]
            df = pd.DataFrame(self.X, columns= self.feature_names)
            df['species'] = self.y
            df['species name'] = species_names
            return df

    def get_Xy_vals(self, df):
        X_vals = df[self.feature_names].values
        y_vals = df['species'].values
        return X_vals, y_vals



    def count_train_test_data(self, train_df, test_df):
        num_train = train_df.shape[0]
        num_test = test_df.shape[0]
        num_outputs = self.label_names.shape[0]
        return num_train, num_test, num_outputs

    def one_hot_encode(self, train_df, num_train_data, test_df, num_test_data, num_outputs):
        y_train = train_df['species']
        y_test = test_df['species']
        y_hot_train = np.zeros((num_train_data, num_outputs))
        y_hot_train[np.arange(num_train_data), y_train] = 1
        y_hot_test = np.zeros((num_test_data, num_outputs))
        y_hot_test[np.arange(num_test_data), y_test] = 1
        return y_hot_train, y_hot_test


class Evaluation(object):
    """docstring for ."""
    def __init__(self, df, y_pred, label_name_dict):
        self.df = df
        self.y_pred = y_pred
        self.label_name_dict = label_name_dict

    def add_pred_classification_to_df(self):
        # It is stupid we have to specify dtype to limit number of allowed chars in string.
        # Find a way around this
        # pred_labels = np.empty(len(self.y_pred), dtype= '|S13')
        pred_labels = np.empty(len(self.y_pred), dtype = int)
        misclassified = np.empty(len(self.y_pred), dtype = bool)
        for numeric_label in self.label_name_dict:
            # pred_labels[np.where(self.y_pred == numeric_label)] = self.label_name_dict[numeric_label]
            pred_labels[np.where(self.y_pred == numeric_label)] = numeric_label
        y_test = self.df['species']
        misclassified[np.where(y_test != self.y_pred)] = True
        indicator = pred_labels
        # indicator[misclassified] = 'misclassified
        indicator[misclassified] = 3
        test_df = self.df
        test_df['predicted'] = self.y_pred
        test_df['indicator'] = indicator
        return test_df

class Plots(object):
    def __init__(self, df):
        self.df = df

    def create_scatter_matrix(self, factor, title, palette = None, color_misclassified= False):
        '''Create a scatter matrix of the variables in df, with differently colored
        points depending on the value of df[factor].
        inputs:
            df: pandas.DataFrame containing the columns to be plotted, as well
                as factor.
            factor: string or pandas.Series. The column indicating which group
                each row belongs to.
            palette: A list of hex codes, at least as long as the number of groups.
                If omitted, a predefined palette will be used, but it only includes
                9 groups.
        '''

        if isinstance(factor, basestring):
            factor_name = factor #save off the name
            factor = self.df[factor] #extract column
            df = self.df.drop(factor_name,axis=1) # remove from df, so it
            # doesn't get a row and col in the plot.

        if palette is None:
            palette = ['#e41a1c', '#377eb8', '#4eae4b',
                        '#994fa1', '#ff8101', '#fdfc33',
                        '#a8572c', '#f482be', '#999999']

        classes = list(set(factor))
        # original_classes = classes

        color_map = dict(zip(classes, palette))

        if len(classes) > len(palette):
            raise ValueError('''Too many groups for the number of colors provided.
                            We only have {} colors in the palette, but you have {}
                            groups.'''.format(len(palette), len(classes)))

        colors = factor.apply(lambda group: color_map[group])
        if color_misclassified:
            if len(classes) + 1 > len(palette):
                raise ValueError('''Please add another color to your palette.
                                We need a unique color for the misclassified values.''')
            colors[np.where(df['misclassified'].values)] = palette[len(classes) + 1] = palette[len(classes) + 1]
            df = df.drop('misclassified', axis=1)

        # unique_class_names = self.df[class_col_name].unique()
        # num_unique_classes = len(unique_class_names)

        title = '{} ('.format(title)

        for i in range(len(classes)):
            title = '{} {} = {},'.format(title, palette[i], classes[i])
        title = '{})'.format(title[:-1])
        scatter_matrix(df, figsize=(25., 25.),
                              marker = '+', c= colors)
        plt.rcParams['axes.labelsize'] = 20 # Is this even working?!?
        plt.suptitle(title, fontsize= 40)
        plt.show()

    def make_radviz(self, class_col_name):
        plt.figure()
        return radviz(self.df, class_col_name)

class Curves(object):
    """docstrin .
    pass"""
    def __init__(self, prob, y, data):
        self.prob = prob
        self.y = y
        self.label_names = data.target_names

    def calculate_threshold_values(self, label_number):
        '''
        Build dataframe of the various confusion-matrix ratios by threshold
        from a list of predicted probabilities and actual y values
        '''

        df = pd.DataFrame({'prob': self.prob[:, label_number], 'y': self.y})
        df.sort_values('prob', inplace=True)

        actual_p = np.sum(df.y == label_number)
        actual_n = df.shape[0] - actual_p

        df['tn'] = (df.y != label_number).cumsum()
        df['fn'] = (df.y == label_number).cumsum()
        df['fp'] = actual_n - df.tn
        df['tp'] = actual_p - df.fn

        df['fpr'] = df.fp/(df.fp + df.tn)
        df['tpr'] = df.tp/(df.tp + df.fn)
        df['precision'] = df.tp/(df.tp + df.fp)
        df['recall'] = df.tp/(df.tp + df.fn)
        df = df.reset_index(drop=True)
        return df



    def plot_roc(self, ax, df, label_number):

        ax.plot([1]+list(df.fpr), [1]+list(df.tpr))
        ax.plot([0,1],[0,1], 'k')
        ax.set_xlabel("fpr")
        ax.set_ylabel("tpr")
        ax.set_title('ROC - {}'.format(self.label_names[label_number]))

    def plot_precision_recall(self, ax, df, label_number):
        ax.plot(df.tpr,df.precision)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title('Precision/Recall - {}'.format(self.label_names[label_number]))
        ax.plot([0,1],[df.precision[0],df.precision[0]], 'k')
        ax.set_xlim(xmin=0,xmax=1.1)
        ax.set_ylim(ymin=0,ymax=1.1)
