import multiprocessing
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import json
import datetime
from multiprocessing import Process, Manager

model_name = "STREAM"

train_set_original = pd.read_csv("./STREAM/data/{}_train_set_original.csv".format(model_name), dtype=np.object)
val_set_original = pd.read_csv("./STREAM/data/{}_val_set_original.csv".format(model_name), dtype=np.object)
test_set_original = pd.read_csv("./STREAM/data/{}_test_set_original.csv".format(model_name), dtype=np.object)

train_set = pd.read_csv("./STREAM/data/{}_train_set.csv".format(model_name))[:1000]      ##
val_set = pd.read_csv("./STREAM/data/{}_val_set.csv".format(model_name))
test_set = pd.read_csv("./STREAM/data/{}_test_set.csv".format(model_name))

def a(i, j, manager_dict):
    dist = []
    for k in range(i, j):
        dist.append(train_set["gpa"][i:i+k].value_counts().to_dict())
    manager_dict[i] = dist


if __name__ == "__main__":
    train_set_gpa_distribution = []
    train_set_gpa_distribution_detail = []

    procs = []
    manager = Manager()
    manager_dict = manager.dict()
    for i in range(8):
        j = train_set["gpa"].size//8*(i+1)
        if i == 7:
            j += train_set["gpa"].size%8
        proc = Process(target=a, args=(train_set["gpa"].size//8*i, j, manager_dict))
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    

    for md in manager_dict:
        print(manager_dict.get(md))
