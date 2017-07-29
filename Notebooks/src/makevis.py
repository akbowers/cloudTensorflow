import matplotlib.pyplot as plt
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
        
    def split_with_labels(self):
        X = self.data.data
        y = self.data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        feature_names = self.data.feature_names
        label_names = self.data.target_names
        species_names = label_names[y]
        test_df = pd.DataFrame(X_test, columns= feature_names)
        test_df['species'] = y_test
        test_df['species name'] = species_names
        return test_df

class Dataframe(object):
    """docstring for ."""
    def __init__(self):
        pass

    def create_df(data):
        y = data.target
        X = data.data
        feature_names = data.feature_names
        label_names = data.target_names
        species_names = label_names[y]
        df = pd.DataFrame(X, columns= feature_names)
        df['species'] = y
        df['species name'] = species_names
        return df
    
    def make_classification_df(X, y, y_pred, label_name_dict):
        # It is stupid we have to specify dtype to limit number of allowed chars in string.
        # Find a way around this 
        pred_labels = np.empty(len(y_pred), dtype= '|S13')
        misclassified = np.empty(len(y_pred), dtype = bool)
        for numeric_label in label_name_dict:
            pred_labels[np.where(y_pred == numeric_label)] = label_name_dict[numeric_label]
        misclassified[np.where(y != y_pred)] = True
        indicator = pred_labels
        indicator[misclassified] = 'misclassified'
        df = pd.DataFrame({'Actual': y, 'Predicted': y_pred, 'Indicator': indicator})
        return df


class Plots(object):
    def __init__(self, df):
        self.df = df

    def create_scatter_matrix(self, class_col_name, class_col_name_numeric, title):
        available_colors = ['red', 'green', 'blue', 'cyan'] # Could give more options here
        colors = []
        unique_class_names = self.df[class_col_name].unique()
        num_unique_classes = len(unique_class_names)
        title = '{} ('.format(title)
        for i in range(num_unique_classes):
            colors.append(available_colors[i])
            title = '{} {} = {},'.format(title, available_colors[i], unique_class_names[i])
        title = '{})'.format(title[:-1])
        scatter_matrix(self.df.drop(class_col_name_numeric, axis=1), figsize=(25., 25.),
                              marker = '+', c= self.df[class_col_name_numeric].apply(lambda x: colors[x]))
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
