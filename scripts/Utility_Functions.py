import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm


class utility_functions:
    def __init__(self):
        pass
    
    @staticmethod
    def MAE(real, predictions):
        # real = (samples x num of sensors)
        # predictions = (predictions on samples x num of sensors)
        return np.mean(np.mean(abs(real-predictions), axis = 0))
    
    @staticmethod
    def MSE(real, predictions):
        # real = (samples x num of sensors)
        # predictions = (predictions on samples x num of sensors)
        return np.mean(np.mean(np.square(abs(real-predictions)),axis = 0))
    
    @staticmethod
    def RMSE(real, predictions):
        # real = (samples x num of sensors)
        # predictions = (predictions on samples x num of sensors)
        return np.mean(np.sqrt(np.mean(np.square(abs(real-predictions)),axis = 0)))
        
    @staticmethod
    def show_barplot(data_list, label, n_bins = 50):
        # data_list = list, data on the plot
        # n_bins = number of bins
        # label = list, labels indicating the data
        for i in range(len(data_list)):
            plt.hist(data_list[i] ,bins=n_bins, label = label[i])
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), 'checkpoint\\images\\barplot.jpg'))
        plt.show()
        
    
    @staticmethod
    def plot_roc_curve(tprs, fprs, n_classes):
        #list of tprs [tpr for class 0, tpr for class 1, ....]
        #list of fprs [fpr for class 0, fpr for class 1, ...]
        for i in range(n_classes):
            plt.plot(fprs[i],tprs[i], label = 'class '+str(i))
            
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), 'checkpoint\\images\\roc_curve.jpg'))
        plt.show()
        
    
    @staticmethod
    def cm2prf(cm):
        # for class 0
        tp0 = cm[0][0]
        fp0 = cm[0][1]
        fn0 = cm[1][0]
        tn0 = cm[1][1]
        pr0 = tp0/(tp0+fp0)
        re0 = tp0/(tp0+fn0)
        f10 = 2*((pr0*re0)/(pr0+re0))

        #for class 1
        tp1 = cm[1][1]
        fp1 = cm[1][0]
        fn1 = cm[0][1]
        tn1 = cm[0][0]
        pr1 = tp1/(tp1+fp1)
        re1 = tp1/(tp1+fn1)
        f11 = 2*((pr1*re1)/(pr1+re1))
        
        return (pr0+pr1)/2 , (re0+re1)/2, (f10+f11)/2 # returning macro average

    