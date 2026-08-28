import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from all_data_tool import load_scaler


WEIGHT_PATH = "../data/model_weight/ae_anomaly.pth"
THRESHOLD_SAVE = "../data/model_weight/ae_threshold.npy"
INPUT_DIM=9

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM,32),
            nn.ReLU(),
            nn.Linear(32,12)
        )
        self.decoder = nn.Sequential(
            nn.Linear(12,32),
            nn.ReLU(),
            nn.Linear(32,INPUT_DIM)
        )
    def forward(self,x):
        z = self.encoder(x)
        rec = self.decoder(z)
        return rec

def train_ae_normal_only(X_normal):
    scaler = load_scaler()
    X_norm = scaler.transform(X_normal)
    Xt = torch.tensor(X_norm,dtype=torch.float32)
    model = AE()
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(),lr=1e-3)
    epochs=80
    for e in range(epochs):
        model.train()
        rec = model(Xt)
        loss = loss_fn(rec,Xt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (e+1)%10==0:
            print(f"AE epoch{e+1},loss:{loss.item():.4f}")
    torch.save(model.state_dict(),WEIGHT_PATH)
    model.eval()
    with torch.no_grad():
        rec_all = model(Xt)
        err = torch.mean((rec_all-Xt)**2,dim=1).numpy()
    threshold = np.percentile(err,95)
    np.save(THRESHOLD_SAVE,threshold)
    print(f"自编码器异常阈值设置为:{threshold:.4f}")
    return model,threshold

def get_ae_model_and_threshold():
    m = AE()
    m.load_state_dict(torch.load(WEIGHT_PATH,weights_only=True))
    m.eval()
    th = float(np.load(THRESHOLD_SAVE))
    return m,th

def detect_anomaly(feature_np:np.ndarray):
    model,threshold = get_ae_model_and_threshold()
    scaler = load_scaler()
    feat_norm = scaler.transform(feature_np.reshape(-1,INPUT_DIM))
    x = torch.tensor(feat_norm,dtype=torch.float32)
    with torch.no_grad():
        rec = model(x)
        err = torch.mean((rec-x)**2,dim=1).numpy()[0]
    is_anomaly = bool(err>threshold)
    return is_anomaly, float(err)

if __name__=="__main__":
    pass
