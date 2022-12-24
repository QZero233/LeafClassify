import torch
from torch import nn
from torch.nn import functional as F

import dataset_process
import train_utils

import pandas as pd
import numpy as np

import torchvision.models as models

device="cuda"


class Residual(nn.Module):
    def __init__(self,input_channel,output_channel,use_1x1=False,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1,stride=stride)
        self.conv2=nn.Conv2d(output_channel,output_channel,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(output_channel)
        self.bn2=nn.BatchNorm2d(output_channel)

        if use_1x1:
            self.conv3=nn.Conv2d(input_channel,output_channel,kernel_size=1,stride=stride)
        else:
            self.conv3=None

    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X=self.conv3(X)
        Y += X
        return F.relu(Y)


def get_net():
    b1=nn.Sequential(
        nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
        nn.BatchNorm2d(64),nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    )
    return nn.Sequential(
        b1,
        Residual(64,64),
        Residual(64,64),
        Residual(64,128,use_1x1=True,stride=2),
        Residual(128,128),
        Residual(128,256,use_1x1=True,stride=2),
        Residual(256,256),
        Residual(256,512,use_1x1=True,stride=2),
        Residual(512,512),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(512,176)
    ).to(device)


# X=torch.rand(size=(1,3,dataset_process.img_size,dataset_process.img_size))
# net=get_net()
# print(net(X))


def k_fold(k,batch_size,lr, epochs, wd):
    batch_count=36
    generator=dataset_process.KFoldDatasetGenerator(k,batch_size,batch_count,device)
    for i in range(k):
        print(f"Start fold {i}")
        train_iter,test_iter=generator.get_data_iter(i)
        plot_name=f"k_cv_fold_{i}_bs_{batch_size}_lr_{lr}_epochs_{epochs}_wd_{wd}"
        net=get_net()
        train_utils.train_for_k_fold(net,train_iter,test_iter,i,lr,epochs,wd,plot_name)


def train(batch_size,lr, epochs, wd):
    batch_count = 36
    generator = dataset_process.KFoldDatasetGenerator(k, batch_size, batch_count, device)
    net=get_net()
    plot_name = f"train_bs_{batch_size}_lr_{lr}_epochs_{epochs}_wd_{wd}"
    train_iter=generator.get_all_data_iter()
    train_utils.train_model(net,train_iter,lr,epochs,wd,plot_name)


def predict(model_name):
    net=get_net()
    net.load_state_dict(torch.load(model_name))
    # load test data
    batch_count=18
    label_map=dataset_process.read_label_map()
    label_indexes=[i for i in range(176)]
    for label in label_map:
        label_indexes[label_map[label]]=label

    result=[]
    for i in range(batch_count):
        file_name=f"test/batch_{i}.bin"
        features=dataset_process.read_all_img_from_file(file_name,device)
        y_hat=net(features).argmax(axis=1)
        for r in y_hat:
            result.append(label_indexes[r])

    df=pd.read_csv("classify-leaves/test.csv")
    df["label"]=result
    df.to_csv("result.csv",index=False)


if __name__ == "__main__":
    k=4
    batch_size=512
    lr=1e-3
    epochs=55
    wd=0
    # k_fold(k,batch_size,lr,epochs,wd)
    # train(batch_size, lr, epochs, wd)
    predict("train_bs_512_lr_0.001_epochs_55_wd_0_epoch_54.params")