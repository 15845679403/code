import os
import sys

# ==========路径修复部分==========
# 获取当前main_api.py所在目录：backend
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录 khr
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
# 1.加入根目录
sys.path.append(ROOT_DIR)
# 2.关键：把algorithm文件夹加入搜索路径，解决 all_data_tool 找不到
ALG_DIR = os.path.join(ROOT_DIR, "algorithm")
sys.path.append(ALG_DIR)
# ==============================

from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from backend.db import get_db, engine
from backend.models import Base, RawVibrationData, FaultDiagnosisRecord, RulPredictRecord, AnomalyAlertRecord

# 下面这些导入现在就可以正常找模块了
from algorithm.all_data_tool import load_csv_signal, signal_filter, sliding_window_sample, extract_all_feature
from algorithm.fault_classify import predict_fault
from algorithm.ae_anomaly import detect_anomaly
from algorithm.lstm_rul import predict_rul, SEQ_LEN

Base.metadata.create_all(bind=engine)
app = FastAPI(title="轴承预测性维护平台API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_SAVE_DIR = os.path.abspath(os.path.join(__file__, "../../data/upload"))
os.makedirs(UPLOAD_SAVE_DIR, exist_ok=True)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """上传振动csv文件，保存文件+写入原始数据表"""
    save_path = os.path.join(UPLOAD_SAVE_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    df = pd.read_csv(save_path)
    total_samples = len(df)
    db_record = RawVibrationData(
        file_name=file.filename,
        sample_count=total_samples,
        file_save_path=save_path
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return {"code": 0, "msg": "上传成功", "raw_data_id": db_record.id}


@app.post("/api/run_diagnose")
async def run_diagnose(raw_data_id: int, db: Session = Depends(get_db)):
    """执行整套算法：故障分类、异常检测、RUL寿命预测，结果入库返回前端"""
    raw_info = db.query(RawVibrationData).filter(RawVibrationData.id == raw_data_id).first()
    if not raw_info:
        return JSONResponse(content={"code": -1, "msg": "数据ID不存在"})
    sig = load_csv_signal(raw_info.file_save_path)
    sig = signal_filter(sig)
    seg_samples = sliding_window_sample(sig, window=256, step=128)
    if len(seg_samples) < SEQ_LEN:
        return {"code":-1,"msg":"数据样本过短，不足以执行LSTM预测"}
    feat_list = []
    for seg in seg_samples:
        feat = extract_all_feature(seg)
        feat_list.append(feat)
    feat_arr = np.array(feat_list)
    #1 故障分类，取第一个片段做预测演示
    fault_name, conf = predict_fault(feat_arr[0])
    fault_db = FaultDiagnosisRecord(
        raw_data_id=raw_data_id,
        fault_type=fault_name,
        confidence=conf
    )
    db.add(fault_db)
    #2 自编码器异常检测
    ano_flag, err = detect_anomaly(feat_arr[0])
    ano_db = AnomalyAlertRecord(
        raw_data_id=raw_data_id,
        is_anomaly=1 if ano_flag else 0,
        recon_error=err
    )
    db.add(ano_db)
    #3 LSTM剩余寿命，取连续SEQ_LEN长度序列
    seq_input = feat_arr[0:SEQ_LEN, :]
    rul_num = predict_rul(seq_input)
    if rul_num <20:
        suggest = "警告：剩余寿命较低，请尽快安排停机检修！"
    elif rul_num<60:
        suggest = "注意，设备逐步老化，密切监测振动状态。"
    else:
        suggest = "设备状态尚可，可以继续运行，定期巡检。"
    rul_db = RulPredictRecord(
        raw_data_id=raw_data_id,
        rul_value=rul_num,
        suggest_maintain=suggest
    )
    db.add(rul_db)
    db.commit()
    return {
        "code":0,
        "data":{
            "fault_type":fault_name,
            "confidence":round(conf,3),
            "is_anomaly":ano_flag,
            "recon_error":round(err,4),
            "rul_value":round(rul_num,2),
            "suggest":suggest
        }
    }


@app.get("/api/get_history")
async def get_history(db:Session=Depends(get_db)):
    """查询全部历史诊断记录"""
    query = db.query(
        RawVibrationData.file_name,
        RawVibrationData.upload_time,
        FaultDiagnosisRecord.fault_type,
        FaultDiagnosisRecord.confidence,
        AnomalyAlertRecord.is_anomaly,
        RulPredictRecord.rul_value,
        RulPredictRecord.suggest_maintain
    ).join(FaultDiagnosisRecord, RawVibrationData.id == FaultDiagnosisRecord.raw_data_id)\
        .join(AnomalyAlertRecord, RawVibrationData.id == AnomalyAlertRecord.raw_data_id)\
        .join(RulPredictRecord, RawVibrationData.id == RulPredictRecord.raw_data_id)\
        .order_by(RawVibrationData.upload_time.desc()).all()
    res = []
    for row in query:
        res.append({
            "filename":row.file_name,
            "upload_time":str(row.upload_time),
            "fault_type":row.fault_type,
            "confidence":row.confidence,
            "is_anomaly":bool(row.is_anomaly),
            "rul_value":row.rul_value,
            "suggest":row.suggest_maintain
        })
    return {"code":0,"list":res}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
