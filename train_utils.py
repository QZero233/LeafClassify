import torch
from torch import nn

from matplotlib import pyplot as plt
import numpy as np


def eval_accuracy(net, test_iter):
    total = 0
    accurate = 0
    for X, y in test_iter:
        y_hat = net(X).argmax(axis=1)

        e = (y == y_hat)
        accurate += e.sum()
        total += len(X)

    return accurate / total


def train_for_k_fold(net, train_iter, test_iter, fold, lr, epochs, wd, plot_name=None):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss = nn.CrossEntropyLoss()
    loss_record = []
    train_accuracy_record = []
    test_accuracy_record = []

    for epoch in range(epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
        loss_record.append(l.sum().item())
        test_accuracy = eval_accuracy(net, test_iter)
        train_accuracy = eval_accuracy(net, train_iter)

        test_accuracy_record.append(test_accuracy.item())
        train_accuracy_record.append(train_accuracy.item())

        print(f"fold {fold}, epoch {epoch}, loss {l}, train accuracy {train_accuracy}, test accuracy {test_accuracy}")

        # save model
        # torch.save(net.state_dict(),plot_name+f"_epoch_{epoch}"+".params")

    # plot
    epoch = np.arange(len(loss_record))
    if plot_name is None:
        plot_name = f"fold_{fold}_lr_{lr}_epochs_{epochs}_wd_{wd}"
    plt.title(plot_name)
    plt.plot(epoch, np.array(loss_record), label="Loss")
    plt.plot(epoch, np.array(test_accuracy_record), label="Test Accuracy")
    plt.plot(epoch, np.array(train_accuracy_record), label="Train Accuracy")
    plt.legend()
    plt.savefig(plot_name+".png")
    # plt.show()


def train_model(net, train_iter,lr, epochs, wd,plot_name=None):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss = nn.CrossEntropyLoss()
    loss_record = []
    train_accuracy_record = []
    for epoch in range(epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
        loss_record.append(l.sum().item())
        train_accuracy = eval_accuracy(net, train_iter)

        train_accuracy_record.append(train_accuracy.item())

        print(f"epoch {epoch}, loss {l}, train accuracy {train_accuracy}")

        # save model
        torch.save(net.state_dict(),plot_name+f"_epoch_{epoch}"+".params")

    # plot
    epoch = np.arange(len(loss_record))
    if plot_name is None:
        plot_name = f"lr_{lr}_epochs_{epochs}_wd_{wd}"
    plt.title(plot_name)
    plt.plot(epoch, np.array(loss_record), label="Loss")
    plt.plot(epoch, np.array(train_accuracy_record), label="Train Accuracy")
    plt.legend()
    plt.savefig(plot_name+".png")
    # plt.show()