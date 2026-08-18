import sys
import traceback
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
import time
import uuid
import threading
import requests
import os
from gpu_probe import GPUProbe
from worker import YOLOWorker, _verify_cuda_or_raise

class ProviderAgent:
    def __init__(self, coordinator_url, price, node_id, provider_id):
        self.coordinator_url = coordinator_url.rstrip('/')
        self.price = price
        self.node_id = node_id or f"GPU-{str(uuid.uuid4())[:8]}"
        self.provider_id = provider_id
        
        self.probe = GPUProbe()
        self.running = True
        self.current_job_id = None
        self.earnings = 0.0
        
    def startup(self):
        # ── GPU pre-flight check ──────────────────────────────────────────
        # This will raise RuntimeError with a clear message if CUDA isn't available,
        # which will propagate to __main__ and prevent registration entirely.
        print("[Agent] Running GPU pre-flight check...")
        _verify_cuda_or_raise()

        metrics = self.probe.get_metrics()
        clean_model = self.probe.get_gpu_model_clean()
        score = self.probe.get_benchmark_score()
        
        print("="*40)
        print(" GPUShare Provider Agent ")
        print("="*40)
        print(f" Node ID:     {self.node_id}")
        print(f" Provider ID: {self.provider_id}")
        print(f" GPU:         {metrics['gpu_model']} (Score: {score})")
        print(f" VRAM:        {metrics['vram_gb']:.1f} GB total")
        print(f" Price:       Rs.{self.price}/hr")
        print("="*40)

    def register(self):
        url = f"{self.coordinator_url}/api/nodes/register"
        metrics = self.probe.get_metrics()
        # Detect local IP to report to coordinator
        import socket as _socket
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "127.0.0.1"

        payload = {
            "gpu_model": metrics['gpu_model'],
            "vram_gb": metrics['vram_gb'],
            "cuda_version": metrics.get('cuda_version', '12.x'),
            "price_per_hour": self.price,
            "ip_address": local_ip,
            "port": 8001,  # provider agent port (not used for jobs, just metadata)
        }

        for i in range(3):
            try:
                response = requests.post(url, json=payload, timeout=5)
                if response.status_code in [200, 201]:
                    data = response.json()
                    # Use coordinator-assigned node_id
                    self.node_id = data.get('node_id', self.node_id)
                    print(f"[Agent] Registered with coordinator. Node ID: {self.node_id}")
                    return True
                else:
                    print(f"[Agent] Registration failed: {response.status_code} {response.text}")
            except Exception as e:
                print(f"[Agent] Registration error: {e}")
            time.sleep(2)

        print("[Agent] Failed to register after 3 attempts.")
        return False

    def unregister(self):
        url = f"{self.coordinator_url}/api/nodes/{self.node_id}/offline"
        try:
            requests.post(url, timeout=5)
            print("[Agent] Node unregistered.")
        except Exception as e:
            print(f"[Agent] Failed to unregister: {e}")

    def heartbeat_loop(self):
        while self.running:
            try:
                metrics = self.probe.get_metrics()
                url = f"{self.coordinator_url}/api/nodes/heartbeat"
                payload = {
                    "node_id": self.node_id,
                    "gpu_utilization": metrics.get("gpu_utilization", 0.0),
                    "vram_used_gb": metrics.get("vram_used_gb", 0.0),
                    "temperature": metrics.get("temperature", 0.0),
                    "power_watts": metrics.get("power_watts", 0.0),
                    "cpu_utilization": metrics.get("cpu_utilization", 0.0),
                    "ram_used_gb": metrics.get("ram_used_gb", 0.0),
                    "network_mbps": metrics.get("network_mbps", 100.0),
                    "available_vram_gb": metrics.get("available_vram_gb", 0.0),
                }
                requests.post(url, json=payload, timeout=5)
            except Exception as e:
                print(f"[Agent] Heartbeat failed: {e}")
            time.sleep(5)

    def poll_for_jobs(self):
        while self.running:
            if not self.current_job_id:
                try:
                    url = f"{self.coordinator_url}/api/nodes/{self.node_id}/pending_job"
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        job = data.get('job')  # coordinator wraps in {"job": {...}}
                        if job:
                            self.start_job(job)
                except Exception as e:
                    pass  # Silently retry
            time.sleep(3)

    def start_job(self, job_data):
        job_id = job_data.get('job_id', 'unknown_job')
        print(f"[Agent] Received new job: {job_id} | model: {job_data.get('model_name')} | epochs: {job_data.get('epochs')}")
        self.current_job_id = job_id

        try:
            requests.post(f"{self.coordinator_url}/api/jobs/{job_id}/start", timeout=5)
        except Exception as e:
            print(f"[Agent] Failed to notify start: {e}")

        def run_worker():
            checkpoint_dir = job_data.get('checkpoint_dir', f"./checkpoints/job_{job_id}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            job_config = {
                "model":             job_data.get('model_name', 'yolov8n.pt'),
                "epochs":            job_data.get('epochs', 10),
                # 'dataset' is the key worker.py reads (was wrongly 'data' before)
                "dataset":           job_data.get('dataset', None),
                "resume_epoch":      job_data.get('resume_epoch', 0),
                "resume_checkpoint": job_data.get('resume_checkpoint'),
            }
            try:
                worker = YOLOWorker(
                    job_config=job_config,
                    checkpoint_dir=checkpoint_dir,
                    coordinator_url=self.coordinator_url,
                    job_id=job_id
                )
            except RuntimeError as e:
                # GPU not available — refuse job cleanly
                print(f"[Agent] Job {job_id} REFUSED — GPU check failed:\n{e}")
                try:
                    requests.post(f"{self.coordinator_url}/api/jobs/{job_id}/interrupt", timeout=5)
                except Exception:
                    pass
                self.current_job_id = None
                return

            try:
                result = worker.run()
                print(f"[Agent] Job {job_id} completed. Output: {result.get('output_path', 'N/A')}")
                try:
                    requests.post(
                        f"{self.coordinator_url}/api/jobs/{job_id}/complete",
                        json={"output_hash": result.get("output_hash", "")},
                        timeout=10
                    )
                except Exception:
                    pass
            except Exception as e:
                print(f"[Agent] Job {job_id} FAILED:\n{traceback.format_exc()}")
                try:
                    requests.post(f"{self.coordinator_url}/api/jobs/{job_id}/interrupt", timeout=5)
                except Exception:
                    pass

            self.current_job_id = None

        threading.Thread(target=run_worker, daemon=True).start()

    def live_metrics_loop(self):
        while self.running:
            time.sleep(10)
            if not self.running:
                break
            metrics = self.probe.get_metrics()
            clean_gpu = self.probe.get_gpu_model_clean()
            status = f"EXECUTING {self.current_job_id}" if self.current_job_id else "IDLE"
            
            print("\n============================")
            print("GPUShare Provider Agent")
            print(f"GPU: {clean_gpu} | Temp: {metrics['temperature']} C")
            print(f"Util: {metrics['gpu_utilization']}% | VRAM: {metrics['vram_used_gb']:.1f}/{metrics['vram_gb']:.1f} GB")
            print(f"Status: {status}")
            print(f"Earnings: Rs.{self.earnings:.2f}")
            print("============================\n")

    def stop(self):
        self.running = False
        self.unregister()
        self.probe.close()
        print(f"[Agent] Shutting down. Total session earnings: Rs.{self.earnings:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator", required=True, help="URL of coordinator")
    parser.add_argument("--price", type=float, default=15.0, help="price per hour in credits")
    parser.add_argument("--node-id", default=None, help="optional node ID")
    parser.add_argument("--provider-id", default="provider_01", help="provider user ID")
    args = parser.parse_args()
    
    agent = ProviderAgent(args.coordinator, args.price, args.node_id, args.provider_id)
    
    try:
        agent.startup()
        if not agent.register():
            sys.exit(1)
            
        t1 = threading.Thread(target=agent.heartbeat_loop, daemon=True)
        t2 = threading.Thread(target=agent.poll_for_jobs, daemon=True)
        t3 = threading.Thread(target=agent.live_metrics_loop, daemon=True)
        
        t1.start()
        t2.start()
        t3.start()
        
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.stop()
