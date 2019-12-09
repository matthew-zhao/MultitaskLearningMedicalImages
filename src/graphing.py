'''
This file is inteded for use outside of Google Compute Engine/AWS to graph results
'''
import matplotlib.pyplot as plt
import argparse
import os

import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

'''
fpr is a np array of false positives
tpr is a np array of true positives
'''
def plot(fpr, tpr, auc, fpr2, tpr2, auc2, fpr3, tpr3, auc3):
    plt.figure()
    lw = 1

    #xnew = np.linspace(fpr.min(),fpr.max(),300)

    #print(fpr.shape, tpr.shape)
    #spl = make_interp_spline(fpr, tpr, k=3) #BSpline object
    #fpr_smooth = spl(xnew)

    # plot the roc curve for the model
    # plt.plot(xnew, fpr_smooth, color='darkorange',
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, marker='v', label='ROC curve multitask FC-only (area = %0.2f)' % auc)
    plt.plot(fpr2, tpr2, color='black',
             lw=lw, marker='^', label='ROC curve multitask DenseBlock + FC (area = %0.2f)' % auc2)
    plt.plot(fpr3, tpr3, color='black',
             lw=lw, marker='<', label='ROC curve single-task (area = %0.2f)' % auc3)

    # plot no skill
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #plt.xlim([0.0, 5.0])
    #plt.ylim([0.0, 5.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Abnormality')
    plt.legend(loc="lower right")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpr', '-f', nargs='+', required=True, help='false positive rate')
    parser.add_argument('--tpr', '-p', nargs='+', required=True, help='true positive rate')
    parser.add_argument('--auc', '-a', help='true positive rate', type=str)
    parser.add_argument('--fpr2', '-f2', nargs='+', required=True, help='false positive rate less sharing')
    parser.add_argument('--tpr2', '-p2', nargs='+', required=True, help='true positive rate less sharing')
    parser.add_argument('--auc2', '-a2', help='true positive rate less sharing', type=str)
    parser.add_argument('--fpr3', '-f3', nargs='+', required=True, help='false positive rate single-task')
    parser.add_argument('--tpr3', '-p3', nargs='+', required=True, help='true positive rate single-task')
    parser.add_argument('--auc3', '-a3', help='true positive rate single-task', type=str)

    args = parser.parse_args()

    func_arguments = {}
    for (key, value) in vars(args).items():
        func_arguments[key] = value

    plot(np.array(sorted(set(func_arguments['fpr']))).astype(np.float), np.array(sorted(set(func_arguments['tpr']))).astype(np.float), float(func_arguments['auc']),
        np.array(sorted(set(func_arguments['fpr2']))).astype(np.float), np.array(sorted(set(func_arguments['tpr2']))).astype(np.float), float(func_arguments['auc2']),
        np.array(sorted(set(func_arguments['fpr3']))).astype(np.float), np.array(sorted(set(func_arguments['tpr3']))).astype(np.float), float(func_arguments['auc3']))

if __name__ == "__main__":
    main()