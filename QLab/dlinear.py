import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = r"D:\QLInternLab\Dlinear.csv" 
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
try:
    data = pd.read_csv(dataset_path)
    if data.empty:
        raise ValueError("The dataset is empty.")
except Exception as e:
    raise RuntimeError(f"Error reading dataset: {e}")
data = data.select_dtypes(include=[np.number])

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

sequence_length = 96
prediction_length = 14
batch_size = 64

class TimeSeriesDataset(Data.Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x, y
train_size = int(len(data_tensor) * 0.8)
train_data, test_data = data_tensor[:train_size], data_tensor[train_size:]

train_dataset = TimeSeriesDataset(train_data, sequence_length, prediction_length)
test_dataset = TimeSeriesDataset(test_data, sequence_length, prediction_length)

train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class DLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(DLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  
        return self.linear(x)
input_size = sequence_length * data.shape[1] 
output_size = prediction_length * data.shape[1]  
model = DLinear(input_size, output_size).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[0], -1)  
            y = y.view(y.shape[0], -1) 

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.6f}")
train_model(model, train_loader, criterion, optimizer, epochs=20)
def evaluate_model(model, test_loader, scaler):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[0], -1) 
            output = model(x)
            predictions.append(output.cpu().numpy())
            actuals.append(y.cpu().numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    
    predictions = scaler.inverse_transform(predictions.reshape(-1, data.shape[1])).reshape(-1, prediction_length, data.shape[1])
    actuals = scaler.inverse_transform(actuals.reshape(-1, data.shape[1])).reshape(-1, prediction_length, data.shape[1])

    return predictions, actuals
predictions, actuals = evaluate_model(model, test_loader, scaler)
mse = mean_squared_error(actuals.flatten(), predictions.flatten())
mae = mean_absolute_error(actuals.flatten(), predictions.flatten())

print(f"Multivariate Forecasting - MSE: {mse:.6f}, MAE: {mae:.6f}")
def plot_results(predictions, actuals, feature_index=-1, num_samples=5):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i+1)
        plt.plot(actuals[i, :, feature_index], label="Actual", marker="o")
        plt.plot(predictions[i, :, feature_index], label="Predicted", marker="x")
        plt.legend()
        plt.title(f"Sample {i+1}")

    plt.tight_layout()
    plt.show()
plot_results(predictions, actuals, feature_index=-1)
