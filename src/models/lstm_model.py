import numpy as np
import torch
import torch.nn as nn

class LSTMMCDropout(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        out = self.fc(last_out)
        return out

class LSTMMCDropoutModel:
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3, 
                 seq_length=10, mc_samples=100):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_length = seq_length
        self.mc_samples = mc_samples
        
        self.model = LSTMMCDropout(input_size, hidden_size, num_layers, dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _create_sequences(self, X, y):
        sequences_X = []
        sequences_y = []
        
        for i in range(len(X) - self.seq_length):
            sequences_X.append(X[i:i+self.seq_length])
            sequences_y.append(y[i+self.seq_length])
            
        return np.array(sequences_X), np.array(sequences_y)
    
    def fit(self, X_train, y_train, epochs=50, batch_size=32, lr=0.001):
        X_seq, y_seq = self._create_sequences(X_train, y_train)
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        return self
    
    def predict_mcdropout(self, X):
        self.model.train()
        
        if len(X) < self.seq_length:
            pad_size = self.seq_length - len(X)
            X_padded = np.vstack([np.zeros((pad_size, X.shape[1])), X])
            X_seq = X_padded[-self.seq_length:].reshape(1, self.seq_length, -1)
        else:
            X_seq = X[-self.seq_length:].reshape(1, self.seq_length, -1)
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                pred = self.model(X_tensor).cpu().numpy().squeeze()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        return predictions
    
    def predict_interval(self, X):
        mc_preds = self.predict_mcdropout(X)
        p10 = np.percentile(mc_preds, 10)
        p50 = np.percentile(mc_preds, 50)
        p90 = np.percentile(mc_preds, 90)
        
        return np.array([[p10, p50, p90]])
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'seq_length': self.seq_length
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        self.seq_length = checkpoint['seq_length']
        
        self.model = LSTMMCDropout(
            self.input_size, 
            self.hidden_size, 
            self.num_layers, 
            self.dropout
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)