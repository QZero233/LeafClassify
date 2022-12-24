import random

import pandas as pd
import numpy as np

import torch
from torch.utils import data

import cv2

import time
import os
import pickle

img_size=224


def read_img_to_numpy(path):
    img = cv2.imread("classify-leaves/"+path)
    img = np.concatenate(
        (img[:, :, 0].reshape((1, img_size, img_size)), img[:, :, 1].reshape((1, img_size, img_size)),
         img[:, :, 2].reshape((1, img_size, img_size))),
        axis=0)
    return img


def read_img_to_tensor(path,device):
    img = read_img_to_numpy(path)
    img = torch.tensor(img, device=device, dtype=torch.float32)
    return img


class KFoldDatasetGenerator:
    def __init__(self,k,batch_size,batch_count,device="cpu"):
        self.k=k
        self.batch_size=batch_size
        self.batch_count=batch_count
        self.device=device

    def get_data_iter(self,fold):
        fold_size=int(self.batch_count/self.k)
        test_start=fold_size*fold
        test_end=min(test_start+fold_size,self.batch_count)

        train_iter=ImageDataLoader(list(range(0,test_start))+list(range(test_end,self.batch_count)),self.batch_size,self.device)
        test_iter=ImageDataLoader(range(test_start,test_end),self.batch_size,self.device)

        return train_iter,test_iter

    def get_all_data_iter(self):
        return ImageDataLoader(range(0,self.batch_count),self.batch_size,self.device)


def append_img_to_file(img_path,file_name):
    img = read_img_to_numpy(img_path).reshape((-1))
    f = open(file_name, "ab+")
    for x in img:
        f.write(x.tobytes())
    f.close()


def read_all_img_from_file(file_name,device):
    size=os.path.getsize(file_name)
    f = open(file_name, "rb")

    result = f.read(size)
    result=torch.frombuffer(result, dtype=torch.uint8).to(device=device, dtype=torch.float32).reshape((-1, 3, img_size, img_size))
    return result


def split_img_into_batches(batch_size):
    train_df = pd.read_csv("classify-leaves/train.csv")

    # format label-to-int map
    label_map = {}
    for label in train_df["label"]:
        if label not in label_map:
            label_map[label] = len(label_map)

    # shuffle
    indexes = list(range(len(train_df)))
    random.shuffle(indexes)

    indexes=indexes[:400]

    labels = []
    img_paths = []
    for i in indexes:
        row = train_df.iloc[i, :]
        img_paths.append(row["image"])
        labels.append(label_map[row["label"]])

    batch_index=0
    processed_image_counter=0
    start_time=time.time()
    for i,img_path in enumerate(img_paths):
        append_img_to_file(img_path,f"data/batch_{batch_index}.bin")
        processed_image_counter += 1

        if (i+1)%batch_size==0 or i==len(img_paths)-1:
            delta=time.time()-start_time

            print(f"Finish batch {batch_index}, processed {processed_image_counter} in {delta} seconds,"
                  f" {processed_image_counter/delta} images per second")

            start_time = time.time()
            processed_image_counter = 0
            batch_index+=1


    # save labels and mapping
    pickle.dump(labels,open("data/labels.dump","wb"))
    pickle.dump(label_map,open("data/label_map.dump","wb"))

    pass


def split_test_set(batch_size):
    df=pd.read_csv("classify-leaves/test.csv")
    img_paths=df["image"].values
    start_time = time.time()
    batch_index=0
    processed_image_counter=0
    for i, img_path in enumerate(img_paths):
        append_img_to_file(img_path, f"test/batch_{batch_index}.bin")
        processed_image_counter += 1

        if (i + 1) % batch_size == 0 or i == len(img_paths) - 1:
            delta = time.time() - start_time

            print(f"Finish batch {batch_index}, processed {processed_image_counter} in {delta} seconds,"
                  f" {processed_image_counter / delta} images per second")

            start_time = time.time()
            processed_image_counter = 0
            batch_index += 1


def read_label_map():
    return pickle.load(open("data/label_map.dump","rb"))


class ImageDataLoader:
    def __init__(self,batch_list,batch_size,device="cpu"):
        self.batch_list=batch_list
        self.batch_size=batch_size
        self.current_batch_index=0
        self.device=device

        # read labels and mapping
        labels=pickle.load(open("data/labels.dump","rb"))
        self.labels=torch.tensor(labels,dtype=torch.int64,device=device)
        self.label_map=pickle.load(open("data/label_map.dump","rb"))

    def __iter__(self):
        self.current_batch_index=0
        return self

    def __next__(self):
        if self.current_batch_index==len(self.batch_list):
            raise StopIteration

        # read batch
        index = self.batch_list[self.current_batch_index]
        labels = self.labels[index * self.batch_size:
                             min((index + 1) * self.batch_size, len(self.labels))]

        start_time=time.time()
        # print(f"Try to read batch {index}")

        file_name = f"data/batch_{index}.bin"
        imgs = read_all_img_from_file(file_name,self.device)
        imgs = torch.tensor(imgs, dtype=torch.float32, device=self.device)

        # increase index
        self.current_batch_index += 1

        end_time=time.time()
        delta=end_time-start_time
        # print(f"Read batch {index} with {len(labels)} samples in {delta} seconds, {len(labels)/delta} samples per second")

        return imgs, labels


if __name__=="__main__":
    split_test_set(512)
    # split_img_into_batches(64)
    # load=ImageDataLoader(range(0,5),2)
    # from matplotlib import pyplot as plt
    # for X,y in load:
    #     print(y)
    #     plt.imshow(X[0,0,:,:])
    #     plt.show()
    # generator=KFoldDatasetGenerator(3,2,5)
    # train,test=generator.get_data_iter(2)
    # for X,y in train:
    #     plt.imshow(X[0,0,:,:])
    #     plt.show()
    #     print(y)
    #
    # print("Test")
    # for X,y in test:
    #     plt.imshow(X[0, 0, :, :])
    #     plt.show()
    #     print(y)
    # print(read_all_img_from_file("data/batch_0.bin","cpu"))

    pass