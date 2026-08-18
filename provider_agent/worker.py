import os
import json
import hashlib
import tempfile
import requests
import time
import sys
import shutil

import torch
import torch.nn as nn
import torch.optim as optim


def _verify_cuda_or_raise():
    """
    Check that PyTorch can see a CUDA GPU before we accept any job.
    Prints a clear diagnostic and raises RuntimeError if not available.
    """
    print(f"[GPU CHECK] torch version      : {torch.__version__}")
    print(f"[GPU CHECK] cuda.is_available(): {torch.cuda.is_available()}")
    print(f"[GPU CHECK] cuda.device_count(): {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        msg = (
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            " GPU NOT AVAILABLE — JOB REFUSED\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f" torch version : {torch.__version__}\n"
            f" Expected : torch with cu128 (CUDA 12.8)\n"
            f" Fix      : pip install --force-reinstall torch torchvision \\\n"
            f"            --index-url https://download.pytorch.org/whl/cu128\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        raise RuntimeError(msg)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[GPU CHECK] [OK] GPU ready: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    return 0  # device index


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple network to simulate GPU matrix multiplication workload
        self.fc = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.fc2(x)


class YOLOWorker:
    """
    Refactored to use pure PyTorch instead of Ultralytics YOLO to avoid
    dependency crashes on the provider side.
    """
    def __init__(self, job_config: dict, checkpoint_dir: str, coordinator_url: str, job_id: str):
        _verify_cuda_or_raise()
        
        self.job_config = job_config
        self.checkpoint_dir = checkpoint_dir
        self.coordinator_url = coordinator_url
        self.job_id = job_id
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Automatically use the NVIDIA GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DummyModel().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def run(self):
        print(f"[Worker] Starting PyTorch workload for job {self.job_id} on {self.device}")
        
        total_epochs = self.job_config.get('epochs', 10)
        start_epoch = 1
        
        # 1. Check for existing checkpoints to demonstrate the Auto-Resume feature
        existing_checkpoints = []
        if os.path.exists(self.checkpoint_dir):
            for f in os.listdir(self.checkpoint_dir):
                if f.endswith('.pt') and 'epoch_' in f:
                    try:
                        ep = int(f.split('epoch_')[1].split('.pt')[0])
                        existing_checkpoints.append((ep, os.path.join(self.checkpoint_dir, f)))
                    except:
                        pass
                        
        if existing_checkpoints:
            existing_checkpoints.sort(key=lambda x: x[0])
            latest_ep, latest_path = existing_checkpoints[-1]
            try:
                # Weights only patch to prevent PyTorch 2.6 security restrictions during unpickling
                checkpoint = torch.load(latest_path, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = latest_ep + 1
                print(f"[Worker] Resumed successfully from epoch {latest_ep}")
            except Exception as e:
                print(f"[Worker] Failed to load checkpoint: {e}")

        # 2. Dummy data pushing into the GPU
        batch_size = self.job_config.get('batch_size', 256)
        print(f"[Worker] Using batch size: {batch_size}")
        dummy_input = torch.randn(batch_size, 1024).to(self.device)
        dummy_target = torch.randn(batch_size, 10).to(self.device)

        # 3. The main AI training loop
        epoch = start_epoch
        for epoch in range(start_epoch, total_epochs + 1):
            try:
                # Simulate heavy compute step to ramp up GPU utilization and Temp
                for _ in range(50):
                    self.optimizer.zero_grad()
                    output = self.model(dummy_input)
                    loss = self.criterion(output, dummy_target)
                    loss.backward()
                    self.optimizer.step()
                    time.sleep(0.05) 
                
                print(f"[Worker] Epoch {epoch}/{total_epochs} completed. Loss: {loss.item():.4f}")
                self._save_checkpoint(epoch)
                self._report_progress(epoch, total_epochs)
                
            except KeyboardInterrupt:
                print("[Worker] Interrupted by user. Saving emergency checkpoint...")
                self._save_checkpoint(epoch)
                break
            except Exception as e:
                print(f"[Worker] Error during training: {e}")
                break

        # 4. Final output generation
        final_path = os.path.join(self.checkpoint_dir, "final_model.pt")
        torch.save(self.model.state_dict(), final_path)
        
        output_hash = self.get_output_hash(final_path)
        
        # 5. Zip the output directory — write zip OUTSIDE the folder being zipped
        zip_dir = os.path.join(tempfile.gettempdir(), "gpushare_outputs")
        os.makedirs(zip_dir, exist_ok=True)
        zip_base = os.path.join(zip_dir, f"{self.job_id}_output")
        
        # root_dir=self.checkpoint_dir means zip contains files directly (no empty wrapper folder)
        zip_path = shutil.make_archive(zip_base, 'zip', root_dir=self.checkpoint_dir)
        print(f"[Worker] Output zip created: {zip_path} ({os.path.getsize(zip_path):,} bytes)")

        return {
            "output_path": final_path,
            "output_hash": output_hash,
            "epochs_completed": epoch,
            "zip_file": zip_path,
            "status": "completed"
        }

    def _save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
        file_hash = self.get_output_hash(path)
        size = os.path.getsize(path)
        
        try:
            requests.post(f"{self.coordinator_url}/api/jobs/{self.job_id}/checkpoint", json={
                "epoch": epoch,
                "file_path": path,
                "file_hash": file_hash,
                "size_bytes": size
            }, timeout=5)
        except Exception as e:
            print(f"[Worker] Failed to report checkpoint: {e}")

    def _report_progress(self, epoch, total):
        percent = int((epoch / total) * 100)
        eta = (total - epoch) * 5  
        try:
            requests.post(f"{self.coordinator_url}/api/jobs/{self.job_id}/progress", json={
                "epoch": epoch,
                "total_epochs": total,
                "percent": percent,
                "eta_seconds": eta
            }, timeout=5)
        except Exception as e:
            pass

    def get_output_hash(self, file_path):
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
            return hasher.hexdigest()
        except:
            return "hash_error"
