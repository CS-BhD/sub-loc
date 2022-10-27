# 混淆矩阵的可视化
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

def plot_confusion_matrix(cm, classes, title='confusion matrix', cmap=plt.cm.Blues):
    """
    plot confusion matrix
    ---------------------
    parmeters:
    mcm:
        multilabel confusion matrix
    classes:
        classes labels
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.array(range(len(classes)))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_multiable_roc_curve(y_label, y_pred, classes):
    """ 
    ROC_curve:
    横坐标:TPR(True positive rate)
    纵坐标:FPR(False positive rate)
    """
    y_label = label_binarize(y_label, classes = [i for i in classes])
    y_pred = label_binarize(y_pred, classes = [i for i in classes])

    # calculate ROC_Curve and ROC_AUC area of all classes
    tpr = dict()
    fpr = dict()
    roc_auc = dict()
    for i in len(classes):
        fpr[i], fpr[i], _ = roc_curve(y_label[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # calculate micro_roc_curve and roc area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # calculate macro_roc_curve and roc area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4) 

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))



    