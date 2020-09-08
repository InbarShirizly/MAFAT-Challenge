from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


# Function for calculating the final ROC-AUC score and plot the ROC curve,
# used in the "Results" section
def stats(pred, actual):
    """
    Computes the model ROC-AUC score and plots the ROC curve.

    Arguments:
    pred -- {ndarray} -- model's probability predictions
    actual -- the true lables

    Returns:
    ROC curve graph and ROC-AUC score
    """
    plt.figure(figsize=(20, 10))
    fpr1, tpr1, _ = roc_curve(actual[0], pred[0])
    fpr2, tpr2, _ = roc_curve(actual[1], pred[1])
    roc_auc = [auc(fpr1, tpr1), auc(fpr2, tpr2)]
    lw = 2
    plt.plot(fpr1, tpr1, lw=lw, label='Training set (ROC-AUC = %0.2f)' % roc_auc[0])
    plt.plot(fpr2, tpr2, lw=lw, label='Validation set (ROC-AUC = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label = 'Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('Training set vs. Validation set ROC curves')
    plt.legend(loc="lower right", prop = {'size': 20})
    plt.show()