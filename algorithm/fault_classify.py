import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from all_data_tool import load_scaler
WEIGHT_PATH = "../data/model_weight/fault_cls.pth"
INPUT_DIM = 9
NUM_CLASS = 4
LABEL_MAP = {0:"正常", 1:"内圈故障", 2:"外圈故障", 3:"滚动体故障"}
class FaultClsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,NUM_CLASS)
        )
    def forward(self,x):
        return self.net(x)
def train_fault_model(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = load_scaler()
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    Xt_train = torch.tensor(X_train,dtype=torch.float32)
    yt_train = torch.tensor(y_train,dtype=torch.long)
    Xt_test = torch.tensor(X_test,dtype=torch.float32)
    model = FaultClsNet()
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(),lr=1e-3)
    epochs=80
    for e in range(epochs):
        model.train()
        pred = model(Xt_train)
        loss = loss_fn(pred,yt_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (e+1)%10==0:
            print(f"FaultCls epoch{e+1},loss:{loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        out_test = model(Xt_test)
        y_pred = torch.argmax(out_test,dim=1).numpy()
    acc = accuracy_score(y_test,y_pred)
    print(f"故障分类测试集准确率:{acc:.4f}")
    print("混淆矩阵:\n",confusion_matrix(y_test,y_pred))
    torch.save(model.state_dict(),WEIGHT_PATH)
    return model
def get_fault_model():
    model = FaultClsNet()
    if os.path.exists(WEIGHT_PATH):
        model.load_state_dict(torch.load(WEIGHT_PATH,weights_only=True))
    model.eval()
    return model
def predict_fault(feature_np:np.ndarray):
    model = get_fault_model()
    scaler = load_scaler()
    feat_norm = scaler.transform(feature_np.reshape(-1,INPUT_DIM))
    x = torch.tensor(feat_norm,dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits,dim=1).numpy()
        pred_idx = np.argmax(prob,axis=1)[0]
        conf = np.max(prob,axis=1)[0]
    return LABEL_MAP[int(pred_idx)], float(conf)
if __name__=="__main__":
    pass
