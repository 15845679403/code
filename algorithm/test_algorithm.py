import numpy as np
from all_data_tool import build_dataset_from_file,fit_and_save_scaler
from fault_classify import train_fault_model,predict_fault
from ae_anomaly import train_ae_normal_only,detect_anomaly
from lstm_rul import train_rul_model,predict_rul

if __name__=="__main__":
    print("=======算法模块单元测试======")

    # ========== 8类数据集配置 ==========
    # 格式：
        # ========== 4类数据集配置（每类2个文件合并）==========
    file_list = [
        ("../data/raw/normal.csv",   0),
        ("../data/raw/98normal.csv", 0),
        ("../data/raw/inner.csv",    1),
        ("../data/raw/106inner.csv", 1),
        ("../data/raw/outer.csv",    2),
        ("../data/raw/131out.csv",   2),
        ("../data/raw/ball.csv",     3),
        ("../data/raw/119ball.csv",  3),
    ]

    

    X_list = []
    y_list = []
    for csv_path, label in file_list:
        feat, lab_arr = build_dataset_from_file(csv_path, label=label)
        X_list.append(feat)
        y_list.append(lab_arr)

    all_feat = np.concatenate(X_list)
    all_label = np.concatenate(y_list)

    # 保存标准化器
    fit_and_save_scaler(all_feat)

    # 训练故障分类模型 (8分类)
    train_fault_model(all_feat, all_label)

    # 取一条样本测试诊断
    fault_name,conf = predict_fault(all_feat[0])
    print(f"故障诊断结果：{fault_name},置信度={conf:.3f}")

    # 自编码器异常检测 ——只用正常样本训练
    normal_feat = X_list[0]
    train_ae_normal_only(normal_feat)
    ano_flag,err = detect_anomaly(all_feat[1])
    print(f"异常检测：是否异常={ano_flag},重构误差={err:.4f}")

    # LSTM‑RUL 剩余寿命预测
    dummy_rul = np.linspace(100, 0, len(all_feat))
    train_rul_model(all_feat, dummy_rul)
    seq_test = all_feat[0:64,:]
    rul_pred = predict_rul(seq_test)
    print(f"LSTM预测剩余寿命：{rul_pred:.2f}")

    print("====全部算法模块运行完成，可等待后端FastAPI调用====")
