import torch
from scipy.io.arff import loadarff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from diffusion import *

import torch
from torch import nn, einsum
import torch.nn.functional as F

from torch.utils.data import DataLoader





def split_classe(x, y, query_class, target_class):
    """
    Split the dataset between the series of query class and series of the target class
    Args:
        x: Dataset
        y: Class attributed to each element in the dataset
        query_class: List of the query class
        target_class: List of the target class

    Returns: the split dataset

    """
    x = x.astype(np.float32)
    query = []
    query_y = []
    target = []
    target_y = []
    for i in range(len(x)):
        if y[i] in query_class:
            query.append(x.values[i])
            query_y.append(y[i]-1)
        elif y[i] in target_class:
            target.append(x.values[i])
            target_y.append(y[i]-1)

    query = np.array(query)
    target = np.array(target)
    query_y = np.array(query_y)
    target_y = np.array(target_y)
    query = torch.tensor(query)
    target = torch.tensor(target)

    print("query : ", query.shape)
    print("target : ", target.shape)
    return query, target, query_y, target_y


def loaddata_arff(dataset_name, batchsize):
    """
    Load a whole .arff dataset
    Args:
        dataset_name: name of the .arff file (Must provide a TRAIN and a TEST file)
        batchsize: Size of the batch
        query_class: List of the query class
        target_class: List of the target class

    Returns: Split datasets between train/test and query/target

    """
    train = loadarff("datasets/"+dataset_name+"_TRAIN.arff")
    train = pd.DataFrame(train[0])
    test = loadarff("datasets/"+dataset_name+"_TEST.arff")
    test = pd.DataFrame(test[0])
    x_train = train.iloc[:, :-1]
    y_train = pd.to_numeric(train.iloc[:, -1], downcast="integer")
    x_test = test.iloc[:, :-1]
    y_test = pd.to_numeric(test.iloc[:, -1], downcast="integer")
    print("Shape of train file : ", x_train.shape)
    print("Shape of test file : ", x_test.shape)


    x_tot=pd.concat([x_train,x_test])
    print("Shape of total file",x_tot.shape)

    y_tot=pd.concat([y_train,y_test])
    print("Shape of total file",y_tot.shape)

    print("\n")



    # Split of the data between the query and target classes
    """
    print("Repartition query/target train")
    query_train, target_train, queryY_train, targetY_train = split_classe(x_train, y_train, query_class, target_class)
    print("Repartition query/target test")
    query_test, target_test, queryY_test, targetY_test = split_classe(x_test, y_test, query_class, target_class)

    print("\n")
    # Dataloader are used to split in batch of the desired batch size
    train_dataloader = torch.utils.data.DataLoader(query_test, batchsize)
    train_target_dataloader = torch.utils.data.DataLoader(target_test, batchsize)
    print("Number of query batch : ", len(train_dataloader))
    print("Number of target batch : ", len(train_target_dataloader))
    """

    """query_train, target_train, queryY_train, targetY_train, query_test, target_test, queryY_test, targetY_test, \
            train_dataloader, train_target_dataloader,"""

    return  x_tot, y_tot


def trans_ts(train,min,max):
  return ((train - min)/(max - min)*2 -1)

def restore_ts(train,min,max):
 return min+(train+1)*(max-min)/2



def main():

    # Load data
    x, y = loaddata_arff("GunPoint", 20)

    v_min, v_max = x.to_numpy().min(), x.to_numpy().max()

    print(x.shape)
    print(y.shape)

    time_serie=x.iloc[[0]].to_numpy()[0]
    print(time_serie)

    fig,ax=plt.subplots(1,5,figsize=(24,3))

    for idx,lv in enumerate([1,config.timesteps//4,config.timesteps//2,config.timesteps*3//4,config.timesteps-1]):

        t = torch.tensor([lv])
        x_start = torch.as_tensor(trans_ts(time_serie,v_min,v_max)).unsqueeze(0)

    
        res_sel=get_noisy_ts(x_start, t)


        ax[idx].plot(restore_ts(res_sel,v_min,v_max))
        ax[idx].set_title("série temporelle bruitée t= %i" % lv)
        
    fig.tight_layout()
    plt.savefig('diffusion_process.png')

    img = torch.randn((568, 1))
    plt.plot( torch.as_tensor(img))
    plt.title("série temporelle pure bruit")
    plt.savefig('série_temporelle_bruit_pure.png')


    #probleme, x isnt a fucking tensor i guess ? jsp 
    dataloader=DataLoader(x, batch_size=config.batch_size, shuffle=True)
    batch = next(iter(dataloader))
    print(batch)



if __name__ == "__main__":
    main()


