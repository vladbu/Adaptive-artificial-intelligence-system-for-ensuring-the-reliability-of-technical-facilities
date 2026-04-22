import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from matplotlib.animation import FuncAnimation
import matplotlib
from sklearn.mixture import GaussianMixture
from eda import get_train_test_tensors
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
import joblib
from matplotlib.animation import FuncAnimation

matplotlib.use('qt5agg')  

x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor = get_train_test_tensors()

gmm = joblib.load('trained_gmm_model.pkl')
scaler = joblib.load('scaler.pkl')

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,

            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2 , output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out
print('параметры модели и оптимизатора...')
# Параметры модели
hidden_dim = 96
num_layers = 1
dropout = 0.2

model = LSTM(
    input_dim=13,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    output_dim=13,
    dropout=dropout
)
model.load_state_dict(torch.load('final_best_weights.pth'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x_test_tensor = x_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

model.eval()

fig1 = plt.figure(figsize=(18, 12))
gs1 = fig1.add_gridspec(3, 3)  # 3 строки, 3 столбца


axes_main = []
for i in range(7):
    row = i // 3
    col = i % 3
    ax = fig1.add_subplot(gs1[row, col])
    axes_main.append(ax)

# Окно 2: GMM и вероятности лучшего кластера
fig2 = plt.figure(figsize=(10, 8))
gs2 = fig2.add_gridspec(2, 1)  # 2 строки, 1 столбец

ax2 = fig2.add_subplot(gs2[0])  # верхний график — гистограмма GMM
ax3 = fig2.add_subplot(gs2[1])  # нижний график — вероятности

# Настройка ax2 (гистограмма кластеров)
bars = ax2.bar(range(gmm.n_components), np.zeros(gmm.n_components))
ax2.set_ylim(0, 1)
ax2.set_xlabel('Кластеры')
ax2.set_ylabel('Вероятность')
ax2.set_xticks(range(gmm.n_components))
ax2.set_xticklabels([
    "Перегрев масла",
    "Межвитковое замыкание в обмотках",
    "Пробой изоляции (между обмотками или на корпус)",
    "Обрыв в обмотках или контактных соединениях",
    "Короткое замыкание (КЗ) на выводах или в сети",
    "Увлажнение изоляции",
    "Повреждение системы охлаждения",
    "Частичные разряды (ЧР) внутри бака",
    "Феррорезонансные явления",
    "Старение и деградация масла"
], rotation=45)

# Настройка ax3 (график вероятности лучшего кластера)
line_best_prod, = ax3.plot([], [], 'g-o', linewidth=2, markersize=6, label='Лучший кластер')
ax3.set_xlabel('Номер образца')
ax3.set_ylabel('Вероятность')
ax3.set_title('Вероятность лучшего кластера для каждого образца')
ax3.grid(True)
ax3.legend()


def create_datetime(sample):
    try:
        # Преобразуем нужные элементы в скаляры через .item()
        year = int(sample[5].item())
        month = int(sample[4].item())
        day = int(sample[3].item())
        hour = int(sample[0].item())
        minute = int(sample[1].item())

        # Ограничим значения в допустимых диапазонах
        year = max(1900, min(2100, year))
        month = max(1, min(12, month))
        day = max(1, min(31, day))
        hour = max(0, min(23, hour))
        minute = max(0, min(59, minute))

        return pd.to_datetime({
            'year': year,
            'month': month,
            'day': day,
            'hour': hour,
            'minute': minute
        })
    except (ValueError, OverflowError):
        # Если ошибка, используем текущую дату
        return pd.Timestamp.now()
datatime_list = []
predictions_7_13_list = []
actual_7_13_list = []
best_probabilities = []
frame_numbers = []  # Исправлено: было frame_numdes

params_labels  = ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
def update(frame):
    sample = x_test_tensor[frame].unsqueeze(0)
    with torch.no_grad():
        output = model(sample)
        predicted_scaled = output.cpu().numpy()

    predicted_unscaled = scaler.inverse_transform(predicted_scaled.reshape(1, -1)).flatten()
    print(f"Frame {frame}: predicted_unscaled[:5] {predicted_unscaled[:5]}")

    current_sample_x = x_test_tensor[frame].cpu().numpy()
    current_datetime = create_datetime(current_sample_x)
    datatime_list.append(current_datetime)

    predictions_7_13 = predicted_unscaled[6:13]
    predictions_7_13_list.append(predictions_7_13)

    actual_scaled = y_test_tensor[frame].cpu().numpy()
    actual_7_13 = scaler.inverse_transform(actual_scaled.reshape(1, -1)).flatten()[6:13]
    actual_7_13_list.append(actual_7_13)

    # Масштабируем предсказания для GMM
    predicted_scaled_for_gmm = scaler.transform(predicted_unscaled.reshape(1, -1))
    prods = gmm.predict_proba(predicted_scaled_for_gmm)
    max_prob = np.max(prods)
    best_cluster = np.argmax(prods)
    print(f"Frame {frame}: Best cluster {best_cluster}, probs {prods[0]}")
    best_probabilities.append(max_prob)
    frame_numbers.append(frame)

    # Обновление Окна 1 (7 основных графиков)
    for i in range(7):
        ax = axes_main[i]
        ax.clear()

        datetimes_i = []
        preds_i = []
        actuals_i = []

        for dt, preds, actuals in zip(datatime_list, predictions_7_13_list, actual_7_13_list):
            datetimes_i.append(dt)
            preds_i.append(preds[i])
            actuals_i.append(actuals[i])

        ax.scatter(datetimes_i, preds_i, c='blue', s=60, alpha=0.6, marker='o', label='Предсказание')
        ax.scatter(datetimes_i, actuals_i, c='red', s=40, alpha=0.6, marker='s', label='Факт')
        ax.set_title(params_labels[i])
        ax.grid(True)

        if i == 0:
            ax.legend()
        else:
            try:
                ax.get_legend().remove()
            except AttributeError:
                pass

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Автомасштабирование по Y
        if preds_i and actuals_i:
            all_vals = preds_i + actuals_i
            ax.set_ylim(min(all_vals) * 0.9, max(all_vals) * 1.1)

    # Обновление Окна 2 (GMM и вероятности)

    # Обновляем гистограмму GMM (ax2)
    for bar, height in zip(bars, prods[0]):
        bar.set_height(height)
    ax2.set_ylim(0, max(prods[0]) * 1.1 if prods[0].size > 0 else 1)
    ax2.set_title(f'Вероятности GMM\nЛучший кластер: {best_cluster}, вероятность: {max_prob:.4f}')

    # Обновляем график вероятностей (ax3)
    ax3.clear()
    ax3.plot(range(len(best_probabilities)), best_probabilities, 'g-o', linewidth=2, markersize=6, label='Лучший кластер')
    ax3.grid(True)
    if len(best_probabilities) > 0:
        ax3.set_ylim(0, max(best_probabilities) * 1.1)

# Создаём анимации для обоих окон
ani1 = FuncAnimation(fig1, update, frames=len(x_test_tensor), repeat=False, blit=False, interval=1500)
ani2 = FuncAnimation(fig2, update, frames=len(x_test_tensor), repeat=False, blit=False, interval=1500)


# Настраиваем layout для обоих окон
plt.figure(fig1.number)
plt.tight_layout()


plt.figure(fig2.number)
plt.tight_layout()


# Показываем оба окна
plt.show()
print("Анимация завершена!")