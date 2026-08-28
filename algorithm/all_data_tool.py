# all_data_tool.py 合并：mat转csv + 数据预处理
import os
import scipy.io
import pandas as pd
import numpy as np
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAT_FOLDER = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "mat_raw"))
OUT_CSV_FOLDER = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw"))
PROC_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))
SCALER_SAVE_PATH = os.path.join(PROC_DATA_DIR, "scaler.pkl")
WINDOW_SIZE = 256

os.makedirs(MAT_FOLDER, exist_ok=True)
os.makedirs(OUT_CSV_FOLDER, exist_ok=True)
os.makedirs(PROC_DATA_DIR, exist_ok=True)

# ===================== mat -> csv 转换部分 =====================
def mat_to_csv(mat_path, out_csv_path):
    mat_dict = scipy.io.loadmat(mat_path)
    de_keys = [k for k in mat_dict.keys() if ("DE_time" in str(k) or "FE_time" in str(k))]
    if len(de_keys) == 0:
        print(f"跳过:{mat_path} 未找到DE_time/FE_time信号")
        return False
    key = de_keys[0]
    signal = mat_dict[key].flatten()
    df = pd.DataFrame({"vibration": signal})
    df.to_csv(out_csv_path, index=False)
    print(f"转换成功 {os.path.basename(mat_path)} -> {os.path.basename(out_csv_path)}")
    return True

def run_mat_convert():
    """执行全部mat文件转csv"""
    file_list = [f for f in os.listdir(MAT_FOLDER) if f.lower().endswith(".mat")]
    print(f"检测到mat文件数量:{len(file_list)}")
    for fname in file_list:
        mat_file = os.path.join(MAT_FOLDER, fname)
        base = os.path.splitext(fname)[0]
        csv_file = os.path.join(OUT_CSV_FOLDER, f"{base}.csv")
        mat_to_csv(mat_file, csv_file)
    print("==== mat全部转换完成 ====\n")

# ===================== 数据预处理部分 =====================
def load_csv_signal(file_path: str) -> np.ndarray:
    df = pd.read_csv(file_path)
    sig = df.iloc[:, 0].dropna().values
    return sig

def signal_filter(signal: np.ndarray, win=5):
    return np.convolve(signal, np.ones(win) / win, mode="same")

def extract_time_feature(sig: np.ndarray):
    mean_val = np.mean(sig)
    var_val = np.var(sig)
    rms = np.sqrt(np.mean(sig ** 2))
    peak = np.max(np.abs(sig))
    kurt = np.mean((sig - mean_val) ** 4) / (var_val ** 2 + 1e-8)
    skew = np.mean((sig - mean_val) ** 3) / ((np.sqrt(var_val)) ** 3 + 1e-8)
    return np.array([mean_val, var_val, rms, peak, kurt, skew])

def extract_freq_feature(sig: np.ndarray):
    fft_sig = np.abs(fft(sig))[:len(sig) // 2]
    return np.array([np.mean(fft_sig), np.var(fft_sig), np.max(fft_sig)])

def extract_all_feature(sig: np.ndarray):
    t_feat = extract_time_feature(sig)
    f_feat = extract_freq_feature(sig)
    return np.concatenate([t_feat, f_feat])

def sliding_window_sample(signal: np.ndarray, window=WINDOW_SIZE, step=128):
    samples = []
    for i in range(0, len(signal) - window, step):
        seg = signal[i:i + window]
        samples.append(seg)
    return np.array(samples)

def build_dataset_from_file(csv_path: str, label=None):
    sig = load_csv_signal(csv_path)
    sig = signal_filter(sig)
    seg_list = sliding_window_sample(sig)
    feat_list = []
    for seg in seg_list:
        feat = extract_all_feature(seg)
        feat_list.append(feat)
    feat_arr = np.array(feat_list)
    if label is not None:
        return feat_arr, np.full(feat_arr.shape[0], label)
    return feat_arr

def fit_and_save_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    return scaler

def load_scaler():
    return joblib.load(SCALER_SAVE_PATH)


if __name__ == "__main__":
    # 先执行mat转csv
    run_mat_convert()
    print("all_data_tool 工具模块加载完毕，可以被其他py文件import调用")
