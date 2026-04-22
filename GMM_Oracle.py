import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from matplotlib.animation import FuncAnimation
import matplotlib
from sklearn.mixture import GaussianMixture
from ChronosCutter import get_train_test_tensors  # Импорт твоей функции из предыдущего файла
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Установка бэкенда для корректного отображения окон анимации (Qt5)
matplotlib.use('qt5agg')  

# 1. ЗАГРУЗКА ДАННЫХ И МОДЕЛЕЙ
x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor = get_train_test_tensors()

# Загружаем предобученную модель кластеризации (GMM) и скалер для денормализации
gmm = joblib.load('trained_gmm_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. АРХИТЕКТУРА МОДЕЛИ LSTM
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Основной слой LSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True # Входные данные: [batch, seq, feature]
        )

        # Полносвязная голова (превращает скрытое состояние в предсказание)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2 , output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # Инициализация скрытых состояний нулями
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        # Берем выход только последнего временного шага (last time step)
        out = self.fc(lstm_out[:, -1, :])
        return out

# 3. ИНИЦИАЛИЗАЦИЯ И ЗАГРУЗКА ВЕСОВ
hidden_dim = 96
num_layers = 1
dropout = 0.2

model = LSTM(input_dim=13, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=13, dropout=dropout)
model.load_state_dict(torch.load('final_best_weights.pth')) # Загрузка обученных весов

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x_test_tensor = x_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)
model.eval() # Перевод в режим оценки (выключает Dropout)

# 4. НАСТРОЙКА ГРАФИКОВ (ОКНО 1: ПАРАМЕТРЫ)
fig1 = plt.figure(figsize=(18, 12))
gs1 = fig1.add_gridspec(3, 3) 
axes_main = []
for i in range(7): # Отрисовываем 7 основных графиков
    ax = fig1.add_subplot(gs1[i // 3, i % 3])
    axes_main.append(ax)

# 5. НАСТРОЙКА ГРАФИКОВ (ОКНО 2: ДИАГНОСТИКА GMM)
fig2 = plt.figure(figsize=(10, 8))
gs2 = fig2.add_gridspec(2, 1)

ax2 = fig2.add_subplot(gs2[0])  # Гистограмма вероятностей кластеров
ax3 = fig2.add_subplot(gs2[1])  # История вероятности лучшего кластера

# Названия дефектов для GMM
cluster_names = [
    "Перегрев масла", "Межвитковое замыкание", "Пробой изоляции", 
    "Обрыв в обмотках", "Короткое замыкание", "Увлажнение изоляции", 
    "Повреждение охл.", "Частичные разряды", "Феррорезонанс", "Старение масла"
]
bars = ax2.bar(range(gmm.n_components), np.zeros(gmm.n_components))
ax2.set_xticks(range(gmm.n_components))
ax2.set_xticklabels(cluster_names, rotation=45, ha='right')

# 6. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
def create_datetime(sample):
    """Извлекает дату и время из признаков входного вектора"""
    try:
        # Предполагается, что в векторе по индексам лежат компоненты даты
        dt = pd.to_datetime({
            'year': int(sample[5].item()), 'month': int(sample[4].item()),
            'day': int(sample[3].item()), 'hour': int(sample[0].item()),
            'minute': int(sample[1].item())
        })
        return dt
    except:
        return pd.Timestamp.now()

# Списки для хранения истории (чтобы графики "росли" во время анимации)
datatime_list, predictions_7_13_list, actual_7_13_list = [], [], []
best_probabilities, frame_numbers = [], []
params_labels = ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']

# 7. ОСНОВНОЙ ЦИКЛ ОБНОВЛЕНИЯ (АНИМАЦИЯ)
def update(frame):
    # Берем один образец и прогоняем через нейросеть
    sample = x_test_tensor[frame].unsqueeze(0)
    with torch.no_grad():
        output = model(sample)
        predicted_scaled = output.cpu().numpy()

    # Денормализация (возвращаем к реальным величинам: градусы, амперы и т.д.)
    predicted_unscaled = scaler.inverse_transform(predicted_scaled.reshape(1, -1)).flatten()
    actual_scaled = y_test_tensor[frame].cpu().numpy()
    actual_unscaled = scaler.inverse_transform(actual_scaled.reshape(1, -1)).flatten()

    # Сохраняем данные для графиков
    current_datetime = create_datetime(x_test_tensor[frame].cpu().numpy())
    datatime_list.append(current_datetime)
    predictions_7_13_list.append(predicted_unscaled[6:13]) # Выбираем нужные индексы параметров
    actual_7_13_list.append(actual_unscaled[6:13])

    # Работа с GMM (Диагностика)
    predicted_scaled_for_gmm = scaler.transform(predicted_unscaled.reshape(1, -1))
    prods = gmm.predict_proba(predicted_scaled_for_gmm) # Вероятности каждого дефекта
    max_prob = np.max(prods)
    best_cluster = np.argmax(prods)
    best_probabilities.append(max_prob)

    # ОБНОВЛЕНИЕ ГРАФИКОВ ПАРАМЕТРОВ
    for i in range(7):
        ax = axes_main[i]
        ax.clear()
        preds_i = [p[i] for p in predictions_7_13_list]
        acts_i = [a[i] for a in actual_7_13_list]
        
        ax.scatter(datatime_list, preds_i, c='blue', s=60, alpha=0.6, label='Предсказание')
        ax.scatter(datatime_list, acts_i, c='red', s=40, alpha=0.6, marker='s', label='Факт')
        ax.set_title(params_labels[i])
        ax.grid(True)
        if i == 0: ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # ОБНОВЛЕНИЕ ГРАФИКОВ ДИАГНОСТИКИ
    # 1. Бар-чарт текущих вероятностей
    for bar, height in zip(bars, prods[0]):
        bar.set_height(height)
    ax2.set_title(f'Текущий диагноз: {cluster_names[best_cluster]} ({max_prob:.2%})')

    # 2. Линейный график уверенности модели
    ax3.clear()
    ax3.plot(best_probabilities, 'g-o', label='Уверенность GMM')
    ax3.set_title('История вероятности выбранного кластера')
    ax3.grid(True)

# ЗАПУСК
ani1 = FuncAnimation(fig1, update, frames=len(x_test_tensor), repeat=False, interval=1500)
ani2 = FuncAnimation(fig2, update, frames=len(x_test_tensor), repeat=False, interval=1500)

plt.show()
