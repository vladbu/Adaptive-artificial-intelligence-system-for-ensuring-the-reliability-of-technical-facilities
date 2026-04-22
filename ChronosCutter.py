import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def create_sequance(length, data):
    """
    Создает выборку по методу скользящего окна.
    
    length: размер 'окна' (сколько шагов назад мы смотрим)
    data: исходный DataFrame
    """
    data = data.values  # Превращаем DataFrame в массив NumPy для скорости
    x, y = [], []
    
    # Проходим циклом по данным, не доходя до конца на размер окна
    for i in range(len(data) - length):
        # x — это кусок данных от i до i + length (наши признаки)
        x.append(data[i:i + length])
        # y — это следующее значение сразу после окна (наша цель/таргет)
        y.append(data[i + length])
        
    return np.array(x), np.array(y)

def get_train_test_tensors(data='data.csv', sequence_length=24, test_size=0.25, random_state=42): 
    # 1. Загрузка данных
    data = pd.read_csv(data)
    
    # 2. Нарезка данных на последовательности (окна)
    x_sequence, y_sequence = create_sequance(sequence_length, data)
    
    # 3. Разделение на обучающую и тестовую выборки
    # shuffle=False критически важно для временных рядов, чтобы не перемешивать хронологию!
    x_train, x_test, y_train, y_test = train_test_split(
        x_sequence, y_sequence, 
        test_size=test_size, 
        random_state=random_state, 
        shuffle=False
    )

    # 4. Масштабирование признаков (X)
    # StandardScaler требует 2D массив, поэтому делаем reshape в "плоский" вид, скалируем, и возвращаем обратно
    scaler_x = StandardScaler()
    x_train_scaled = scaler_x.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_test_scaled = scaler_x.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

    # 5. Масштабирование целевой переменной (Y)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # 6. Конвертация в тензоры PyTorch для загрузки в нейросеть
    x_train_tensor = torch.FloatTensor(x_train_scaled)
    x_test_tensor = torch.FloatTensor(x_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    return x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor
