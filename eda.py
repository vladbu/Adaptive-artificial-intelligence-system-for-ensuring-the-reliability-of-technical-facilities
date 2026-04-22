import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def create_sequance(length, data):
    
    data = data.values
    x,y = [],[]
    for i in range(len(data) - length):
        x.append(data[i:i + length])
        y.append(data[i + length])
    return np.array(x) , np.array(y)

def get_train_test_tensors(data = 'data.csv' , sequence_length = 24,  test_size=0.25, random_state=42): 
    data = pd.read_csv(data)
    x_sequence, y_sequence = create_sequance(sequence_length, data)
    x_train, x_test, y_train, y_test = train_test_split(x_sequence, y_sequence, test_size=test_size, random_state=random_state,shuffle=False)

    scaler_x = StandardScaler()
    x_train_scaled = scaler_x.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_test_scaled = scaler_x.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)


    x_train_tensor = torch.FloatTensor(x_train_scaled)
    x_test_tensor = torch.FloatTensor(x_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    return x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor
