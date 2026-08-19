import asyncio
import json
import socket
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

import os
import shutil
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import engine, SessionLocal, Base, create_tables, get_db
from database import User, GpuNode, NodeMetric, Job, Checkpoint, Transaction
from gpumatch import compute_gpumatch_score, predict_availability, GPU_BENCHMARK_TABLE

def get_lan_ip() -> str:
    """Get this machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

app = FastAPI(title="GPUShare Coordinator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        msg_str = json.dumps(message)
        for connection in self.active_connections:
            try:
                await connection.send_text(msg_str)
            except Exception:
                pass

manager = ConnectionManager()

def seed_db():
    db = SessionLocal()
    try:
        if not db.query(User).filter_by(user_id="consumer_01").first():
            consumer = User(user_id="consumer_01", name="You", role="consumer", credits=100.0)
            db.add(consumer)
        if not db.query(User).filter_by(user_id="provider_01").first():
            provider = User(user_id="provider_01", name="Friend's RTX 4070", role="provider", credits=0.0)
            db.add(provider)
        db.commit()
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    create_tables()
    seed_db()
    asyncio.create_task(heartbeat_monitor())
    lan_ip = get_lan_ip()
    print("\n" + "="*52)
    print("  GPUShare Coordinator ONLINE")
    print(f"  LAN IP  : {lan_ip}")
    print(f"  API URL : http://{lan_ip}:8000")
    print(f"  Docs    : http://{lan_ip}:8000/docs")
    print("  Tell your friend to use this IP address!")
    print("="*52 + "\n")


async def heartbeat_monitor():
    while True:
        try:
            db = SessionLocal()
            cutoff = datetime.utcnow() - timedelta(seconds=20)
            stale_nodes = db.query(GpuNode).filter(GpuNode.last_heartbeat < cutoff, GpuNode.status == "online").all()
            
            for node in stale_nodes:
                node.status = "offline"
                
                # Check for active jobs on this node
                active_jobs = db.query(Job).filter(Job.node_id == node.node_id, Job.status == "EXECUTING").all()
                for job in active_jobs:
                    job.status = "INTERRUPTED"
                    await manager.broadcast({
                        "event": "job_interrupted",
                        "job_id": job.job_id,
                        "node_id": node.node_id
                    })
                
                await manager.broadcast({
                    "event": "node_offline",
                    "node_id": node.node_id
                })
            
            if stale_nodes:
                db.commit()
            
            db.close()
        except Exception as e:
            print(f"Heartbeat monitor error: {e}")
        
        await asyncio.sleep(10)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/")
def root():
    return {"status": "ok", "version": "1.0"}

@app.get("/api/status")
def get_status(db: Session = Depends(get_db)):
    online_nodes = db.query(GpuNode).filter(GpuNode.status == "online").count()
    active_jobs = db.query(Job).filter(Job.status.in_(["QUEUED", "EXECUTING"])).count()
    return {
        "status": "online",
        "online_nodes_count": online_nodes,
        "active_jobs": active_jobs
    }

class NodeRegisterReq(BaseModel):
    gpu_model: str
    vram_gb: float
    cuda_version: str
    price_per_hour: float
    ip_address: str
    port: int

@app.post("/api/nodes/register")
async def register_node(req: NodeRegisterReq, db: Session = Depends(get_db)):
    node_id = f"node_{uuid.uuid4().hex[:8]}"
    benchmark = GPU_BENCHMARK_TABLE.get(req.gpu_model, 50.0)
    
    node = GpuNode(
        node_id=node_id,
        gpu_model=req.gpu_model,
        vram_gb=req.vram_gb,
        cuda_version=req.cuda_version,
        benchmark_score=benchmark,
        price_per_hour=req.price_per_hour,
        ip_address=req.ip_address,
        port=req.port,
        status="online",
        last_heartbeat=datetime.utcnow()
    )
    db.add(node)
    db.commit()
    
    await manager.broadcast({"event": "node_registered", "node_id": node_id})
    return {"node_id": node_id, "status": "online"}

class HeartbeatReq(BaseModel):
    node_id: str
    gpu_utilization: float
    vram_used_gb: float
    temperature: float
    power_watts: float
    cpu_utilization: float
    ram_used_gb: float
    network_mbps: float
    available_vram_gb: float

@app.post("/api/nodes/heartbeat")
def node_heartbeat(req: HeartbeatReq, db: Session = Depends(get_db)):
    """Sync endpoint — FastAPI runs this in a thread pool, never blocks the event loop."""
    node = db.query(GpuNode).filter(GpuNode.node_id == req.node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    node.last_heartbeat = datetime.utcnow()
    node.status = "online"
    metric = NodeMetric(
        node_id=req.node_id,
        gpu_utilization=req.gpu_utilization,
        vram_used_gb=req.vram_used_gb,
        temperature=req.temperature,
        power_watts=req.power_watts,
        cpu_utilization=req.cpu_utilization,
        ram_used_gb=req.ram_used_gb,
        network_mbps=req.network_mbps,
        available_vram_gb=req.available_vram_gb
    )
    db.add(metric)
    db.commit()
    return {"status": "ok"}

@app.get("/api/nodes")
def list_nodes(db: Session = Depends(get_db)):
    nodes = db.query(GpuNode).all()
    result = []
    for n in nodes:
        latest = db.query(NodeMetric).filter(NodeMetric.node_id == n.node_id).order_by(NodeMetric.timestamp.desc()).first()
        result.append({
            "node_id": n.node_id,
            "status": n.status,
            "gpu_model": n.gpu_model,
            "vram_gb": n.vram_gb,
            "price_per_hour": n.price_per_hour,
            "trust_score": n.trust_score,
            "latest_metrics": {
                "gpu_utilization": latest.gpu_utilization if latest else 0,
                "available_vram_gb": latest.available_vram_gb if latest else n.vram_gb,
                "temperature": latest.temperature if latest else 0
            }
        })
    return result

@app.get("/api/nodes/{node_id}")
def get_node(node_id: str, db: Session = Depends(get_db)):
    n = db.query(GpuNode).filter(GpuNode.node_id == node_id).first()
    if not n:
        raise HTTPException(status_code=404, detail="Node not found")
    return n

class JobSubmitReq(BaseModel):
    consumer_id: str
    workload_type: str
    model_name: str
    vram_required_gb: float
    max_budget: float
    estimated_minutes: int
    batch_size: Optional[int] = 256
    input_hash: Optional[str] = None

@app.post("/api/jobs/submit")
async def submit_job(req: JobSubmitReq, db: Session = Depends(get_db)):
    # Run GPUMatch
    nodes = db.query(GpuNode).filter(GpuNode.status == "online").all()
    best_match = None
    best_score = -1.0
    match_details = {}

    job_dict = {
        "vram_required_gb": req.vram_required_gb,
        "estimated_minutes": req.estimated_minutes
    }

    for n in nodes:
        node_dict = {
            "vram_gb": n.vram_gb,
            "status": n.status,
            "gpu_model": n.gpu_model,
            "trust_score": n.trust_score,
            "ip_address": n.ip_address,
            "port": n.port,
            "price_per_hour": n.price_per_hour
        }
        latest = db.query(NodeMetric).filter(NodeMetric.node_id == n.node_id).order_by(NodeMetric.timestamp.desc()).first()
        metric_dict = {"available_vram_gb": latest.available_vram_gb} if latest else {}
        
        score_res = compute_gpumatch_score(node_dict, job_dict, metric_dict)
        if score_res["eligible"] and score_res["final_score"] > best_score:
            best_score = score_res["final_score"]
            best_match = n
            match_details = score_res

    if not best_match:
        raise HTTPException(status_code=400, detail="No suitable node found")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    
    job = Job(
        job_id=job_id,
        consumer_id=req.consumer_id,
        node_id=best_match.node_id,
        workload_type=req.workload_type,
        model_name=req.model_name,
        vram_required_gb=req.vram_required_gb,
        max_budget=req.max_budget,
        batch_size=req.batch_size,
        status="QUEUED",
        gpumatch_score=best_score,
        expected_cost=match_details.get("expected_cost"),
        nominal_cost=match_details.get("nominal_cost"),
        input_hash=req.input_hash,
        estimated_minutes=req.estimated_minutes
    )
    db.add(job)
    db.commit()

    await manager.broadcast({
        "event": "job_submitted",
        "job_id": job_id,
        "node_id": best_match.node_id
    })
    
    return {"job_id": job_id, "status": "QUEUED", "matched_node": best_match.node_id, "score": best_score}

@app.get("/api/jobs")
def list_jobs(db: Session = Depends(get_db)):
    return db.query(Job).all()

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    j = db.query(Job).filter(Job.job_id == job_id).first()
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    
    checkpoints = db.query(Checkpoint).filter(Checkpoint.job_id == job_id).all()
    return {
        "job": j,
        "checkpoints": checkpoints
    }

@app.post("/api/jobs/{job_id}/start")
async def start_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.status = "EXECUTING"
    job.started_at = datetime.utcnow()
    db.commit()
    await manager.broadcast({"event": "job_started", "job_id": job_id})
    return {"status": "EXECUTING"}

class ProgressReq(BaseModel):
    epoch: int
    percent: float
    eta_seconds: int

@app.post("/api/jobs/{job_id}/progress")
async def job_progress(job_id: str, req: ProgressReq, db: Session = Depends(get_db)):
    await manager.broadcast({
        "event": "job_progress",
        "job_id": job_id,
        "epoch": req.epoch,
        "percent": req.percent,
        "eta_seconds": req.eta_seconds
    })
    return {"status": "ok"}

class CheckpointReq(BaseModel):
    epoch: int
    file_path: str
    file_hash: str
    size_bytes: int

@app.post("/api/jobs/{job_id}/checkpoint")
def save_checkpoint(job_id: str, req: CheckpointReq, db: Session = Depends(get_db)):
    chk = Checkpoint(
        job_id=job_id,
        epoch=req.epoch,
        file_path=req.file_path,
        file_hash=req.file_hash,
        size_bytes=req.size_bytes
    )
    db.add(chk)
    db.commit()
    return {"status": "ok"}

class JobCompleteReq(BaseModel):
    output_hash: str

@app.post("/api/jobs/{job_id}/upload_output")
def upload_job_output(job_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    file_path = os.path.join(artifacts_dir, f"{job_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    job.output_file_path = file_path
    db.commit()
    return {"status": "ok", "file_path": file_path}

@app.get("/api/jobs/{job_id}/download_output")
async def download_job_output(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if not job.output_file_path or not os.path.exists(job.output_file_path):
        raise HTTPException(status_code=404, detail="Output file not found for this job")
        
    return FileResponse(
        path=job.output_file_path,
        filename=os.path.basename(job.output_file_path),
        media_type="application/zip"
    )

@app.post("/api/jobs/{job_id}/complete")
async def complete_job(job_id: str, req: JobCompleteReq, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.completed_at = datetime.utcnow()
    job.output_hash = req.output_hash
    if job.started_at:
        job.compute_seconds = (job.completed_at - job.started_at).total_seconds()
    else:
        job.compute_seconds = 0
    job.verified = bool(job.input_hash and req.output_hash)
    node = db.query(GpuNode).filter(GpuNode.node_id == job.node_id).first()
    if job.verified and node:
        job.actual_cost = (job.compute_seconds / 3600.0) * node.price_per_hour
        consumer = db.query(User).filter(User.user_id == job.consumer_id).first()
        provider = db.query(User).filter(User.user_id == "provider_01").first()
        if consumer and provider:
            consumer.credits -= job.actual_cost
            provider_cut = job.actual_cost * 0.90
            platform_fee = job.actual_cost * 0.10
            provider.credits += provider_cut
            txn = Transaction(
                job_id=job.job_id,
                provider_id=provider.user_id,
                consumer_id=consumer.user_id,
                amount=job.actual_cost,
                fee=platform_fee,
                provider_earns=provider_cut,
                status="COMPLETED"
            )
            db.add(txn)
        node.trust_score = min(100.0, node.trust_score + 2.0)
        node.jobs_completed += 1
        node.jobs_total += 1
    elif node:
        node.trust_score = max(0.0, node.trust_score - 5.0)
        node.jobs_total += 1
    job.status = "COMPLETED"
    db.commit()
    await manager.broadcast({"event": "job_completed", "job_id": job_id})
    return {"status": "COMPLETED", "verified": job.verified}

@app.post("/api/jobs/{job_id}/interrupt")
async def interrupt_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.status = "INTERRUPTED"
    db.commit()
    await manager.broadcast({"event": "job_interrupted", "job_id": job_id})
    return {"status": "INTERRUPTED"}

@app.get("/api/jobs/{job_id}/checkpoints")
def list_checkpoints(job_id: str, db: Session = Depends(get_db)):
    return db.query(Checkpoint).filter(Checkpoint.job_id == job_id).all()

@app.get("/api/billing/balance")
def get_balance(db: Session = Depends(get_db)):
    c = db.query(User).filter(User.user_id == "consumer_01").first()
    p = db.query(User).filter(User.user_id == "provider_01").first()
    return {
        "consumer_credits": c.credits if c else 0,
        "provider_credits": p.credits if p else 0
    }

@app.get("/api/billing/transactions")
def list_transactions(db: Session = Depends(get_db)):
    return db.query(Transaction).all()

@app.get("/api/metrics/history/{node_id}")
def get_metrics_history(node_id: str, db: Session = Depends(get_db)):
    return db.query(NodeMetric).filter(NodeMetric.node_id == node_id).order_by(NodeMetric.timestamp.desc()).limit(50).all()

# ── Provider Agent Polling Endpoint ────────────────────────────────────────
@app.get("/api/nodes/{node_id}/pending_job")
def get_pending_job(node_id: str, db: Session = Depends(get_db)):
    """Provider agent polls this every 3s to get its next assigned job."""
    node = db.query(GpuNode).filter(GpuNode.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    job = (
        db.query(Job)
        .filter(Job.node_id == node_id, Job.status == "QUEUED")
        .order_by(Job.submitted_at)
        .first()
    )
    if not job:
        return {"job": None}

    # Get latest checkpoint for resume support
    latest_chk = (
        db.query(Checkpoint)
        .filter(Checkpoint.job_id == job.job_id)
        .order_by(Checkpoint.epoch.desc())
        .first()
    )

    return {
        "job": {
            "job_id": job.job_id,
            "workload_type": job.workload_type,
            "model_name": job.model_name or "yolov8n.pt",
            "epochs": max(10, (job.estimated_minutes or 10) // 2),
            "data": "coco8.yaml",
            "checkpoint_dir": f"./checkpoints/job_{job.job_id}",
            "resume_epoch": latest_chk.epoch if latest_chk else 0,
            "resume_checkpoint": latest_chk.file_path if latest_chk else None,
            "vram_required_gb": job.vram_required_gb,
            "batch_size": job.batch_size,
        }
    }

# ── Node Offline / Graceful Shutdown ───────────────────────────────────────
@app.post("/api/nodes/{node_id}/offline")
async def node_offline(node_id: str, db: Session = Depends(get_db)):
    """Called by provider agent on graceful shutdown."""
    node = db.query(GpuNode).filter(GpuNode.node_id == node_id).first()
    if node:
        node.status = "offline"
        db.commit()
        await manager.broadcast({"event": "node_offline", "node_id": node_id})
    return {"status": "offline"}

# ── Coordinator entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
