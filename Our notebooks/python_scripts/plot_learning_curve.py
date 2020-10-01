import matplotlib.pyplot as plt
import numpy as np

def plot_loss_and_accuracy_over_epoches(history):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_title("Loss over epoches")
    ax1.grid()
    legend = ax1.legend(loc='best', shadow=True)

    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
    ax2.set_title("accuracy over epoches")
    ax2.grid()


    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()