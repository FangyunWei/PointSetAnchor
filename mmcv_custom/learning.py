import os
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(train_loss, valid_loss, log_path, jobName):
    '''
    Use matplotlib to plot learning curve at the end of training
    train_loss & valid_loss must be 'list' type
    '''
    fig = plt.figure(figsize=(12, 5))
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel(jobName)

    epochs = np.arange(len(train_loss))
    plt.plot(epochs, np.array(train_loss), 'r', label='train')
    plt.plot(epochs, np.array(valid_loss), 'b', label='valid')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(log_path, jobName + '.png'))
    plt.cla()
    plt.close(fig)
