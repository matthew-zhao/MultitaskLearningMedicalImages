'''
This file is inteded for use outside of Google Compute Engine/AWS to graph results
'''
import matplotlib.pyplot as plt
import argparse

'''
fpr is a np array of false positives
tpr is a np array of true positives
'''
def plot(fpr, tpr, auc):
    plt.figure()
    lw = 2
    # plot the roc curve for the model
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, marker='.', label='ROC curve (area = %0.2f)' % auc)
    # plot no skill
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Abnormality')
    plt.legend(loc="lower right")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpr', '-f', nargs='+', required=True, help='false positive rate')
    parser.add_argument('--tpr', '-p', nargs='+', required=True, help='true positive rate')
    parser.add_argument('--auc', '-a', default=1.0, help='true positive rate')

    args = parser.parse_args()

    func_arguments = {}
    for (key, value) in vars(args).items():
        func_arguments[key] = value

    plot(func_arguments['fpr'], func_arguments['tpr'], func_arguments['auc'])

if __name__ == "main":
    main()