from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from backend.db import Base

class RawVibrationData(Base):
    __tablename__ = "raw_vibration_data"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    upload_time = Column(DateTime, server_default=func.now())
    sample_count = Column(Integer)
    file_save_path = Column(String(500))

class FaultDiagnosisRecord(Base):
    __tablename__ = "fault_diagnosis_record"
    id = Column(Integer, primary_key=True)
    raw_data_id = Column(Integer)
    fault_type = Column(String(50))
    confidence = Column(Float)
    diagnose_time = Column(DateTime, server_default=func.now())

class RulPredictRecord(Base):
    __tablename__ = "rul_predict_record"
    id = Column(Integer, primary_key=True)
    raw_data_id = Column(Integer)
    rul_value = Column(Float)
    suggest_maintain = Column(Text)
    predict_time = Column(DateTime, server_default=func.now())

class AnomalyAlertRecord(Base):
    __tablename__ = "anomaly_alert_record"
    id = Column(Integer, primary_key=True)
    raw_data_id = Column(Integer)
    is_anomaly = Column(Integer)
    recon_error = Column(Float)
    alert_time = Column(DateTime, server_default=func.now())
