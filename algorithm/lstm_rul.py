import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from all_data_tool import load_scaler


WEIGHT_PATH = "../data/model_weight/lstm_rul.pth"
SEQ_LEN = 64
FEAT_DIM =9

class LSTM_RUL(nn.Module):
    def __init__(self,input_size=FEAT_DIM,hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x):
        out,_ = self.lstm(x)
        last_step = out[:,-1,:]
        return self.fc(last_step)

def create_seq_data(X_feat, rul_label, seq_len=SEQ_LEN):
    seqs, ys = [],[]
    for i in range(len(X_feat)-seq_len):
        seqs.append(X_feat[i:i+seq_len,:])
        ys.append(rul_label[i+seq_len])
    return np.array(seqs), np.array(ys)

def train_rul_model(X_all, rul_all):
    scaler = load_scaler()
    X_norm = scaler.transform(X_all)
    X_seq,y_rul = create_seq_data(X_norm,rul_all,SEQ_LEN)
    X_train,X_test,y_train,y_test = train_test_split(X_seq,y_rul,test_size=0.2,random_state=42)

    Xt_train = torch.tensor(X_train,dtype=torch.float32)
    yt_train = torch.tensor(y_train,dtype=torch.float32).unsqueeze(-1)

    model = LSTM_RUL()
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(),lr=1e-3)
    epochs=100
    for e in range(epochs):
        model.train()
        pred = model(Xt_train)
        loss = loss_fn(pred,yt_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (e+1)%20==0:
            print(f"LSTM RUL epoch{e+1},loss:{loss.item():.4f}")
    torch.save(model.state_dict(),WEIGHT_PATH)
    return model

def get_rul_model():
    m = LSTM_RUL()
    if os.path.exists(WEIGHT_PATH):
        m.load_state_dict(torch.load(WEIGHT_PATH,weights_only=True))
    m.eval()
    return m

def predict_rul(seq_feature_np:np.ndarray):
    model = get_rul_model()
    scaler = load_scaler()
    feat_norm = scaler.transform(seq_feature_np)
    x = torch.tensor(feat_norm[np.newaxis,...],dtype=torch.float32)
    with torch.no_grad():
        pred = model(x)
    return float(pred.squeeze().item())

if __name__=="__main__":
    pass
