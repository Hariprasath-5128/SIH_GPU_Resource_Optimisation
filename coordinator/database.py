import os
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

DATABASE_URL = "sqlite:///./coordinator.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True, index=True)
    name = Column(String)
    role = Column(String)
    credits = Column(Float, default=100.0)
    reputation = Column(Float, default=80.0)

class GpuNode(Base):
    __tablename__ = "gpu_nodes"
    node_id = Column(String, primary_key=True, index=True)
    gpu_model = Column(String)
    vram_gb = Column(Float)
    cuda_version = Column(String)
    benchmark_score = Column(Float)
    price_per_hour = Column(Float)
    trust_score = Column(Float, default=80.0)
    jobs_completed = Column(Integer, default=0)
    jobs_total = Column(Integer, default=0)
    uptime_seconds = Column(Integer, default=0)
    status = Column(String, default='offline')
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String)
    port = Column(Integer)

class NodeMetric(Base):
    __tablename__ = "node_metrics"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    node_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    gpu_utilization = Column(Float)
    vram_used_gb = Column(Float)
    temperature = Column(Float)
    power_watts = Column(Float)
    cpu_utilization = Column(Float)
    ram_used_gb = Column(Float)
    network_mbps = Column(Float)
    available_vram_gb = Column(Float)

class Job(Base):
    __tablename__ = "jobs"
    job_id = Column(String, primary_key=True, index=True)
    consumer_id = Column(String)
    node_id = Column(String, nullable=True)
    workload_type = Column(String)
    model_name = Column(String)
    vram_required_gb = Column(Float)
    max_budget = Column(Float)
    status = Column(String)
    gpumatch_score = Column(Float, nullable=True)
    expected_cost = Column(Float, nullable=True)
    nominal_cost = Column(Float, nullable=True)
    actual_cost = Column(Float, nullable=True)
    input_hash = Column(String, nullable=True)
    output_hash = Column(String, nullable=True)
    output_file_path = Column(String, nullable=True)
    verified = Column(Boolean, default=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    compute_seconds = Column(Float, nullable=True)
    checkpoint_interval = Column(Integer, default=5)
    estimated_minutes = Column(Integer, nullable=True)

class Checkpoint(Base):
    __tablename__ = "checkpoints"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    job_id = Column(String, index=True)
    epoch = Column(Integer)
    file_path = Column(String)
    file_hash = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    size_bytes = Column(Integer)

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    job_id = Column(String, index=True)
    provider_id = Column(String)
    consumer_id = Column(String)
    amount = Column(Float)
    fee = Column(Float)
    provider_earns = Column(Float)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
