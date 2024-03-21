import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support

def fnPrecision_Recall_Curve_Plot(y_test, pred_proba, plot_flag = True):
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba)
    
    optimal_thresholds = sorted(list(zip(np.abs(precisions - recalls), thresholds)), 
                                key = lambda x: x[0], reverse = False)[0][1]
    
    if plot_flag:
        
        threshold_index = thresholds.shape[0]
        
        plt.figure(figsize=(8,6))
        
        plt.plot(thresholds, precisions[0:threshold_index],'r--',label='precision')
        plt.plot(thresholds, recalls[0:threshold_index],label='recall')
        
        start,end = plt.xlim()
        
        plt.xticks(np.around(np.arange(start,end, 0.1), 2))
        plt.xlabel('Threshold value')
        plt.ylabel('precision and recall value')
        plt.title('Precision-Recall vs. Threshold Curve')
        plt.legend()
        plt.grid()
        
        plt.show()
    
    return optimal_thresholds