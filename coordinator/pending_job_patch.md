# Pending Job Endpoint Patch
Add to main.py in the nodes router:

```python
@app.get("/api/nodes/{node_id}/pending_job")
async def get_pending_job(node_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(
        Job.node_id == node_id,
        Job.status == 'QUEUED'
    ).order_by(Job.submitted_at).first()
    if job:
        return {"job": {"job_id": job.job_id, "workload_type": job.workload_type, "model_name": job.model_name, "epochs": job.checkpoint_interval * 2, "data": "coco8.yaml", "checkpoint_dir": f"./checkpoints/job_{job.job_id}"}}
    return {"job": None}
```
