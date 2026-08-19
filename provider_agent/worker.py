import time
import os
import hashlib
import requests
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        five_gb = 5 * 1024 * 1024 * 1024
        fraction = min(five_gb / total_memory, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        print(f"[Worker] VRAM Restricted to a maximum of 5.0 GB ({fraction*100:.1f}%)")
    except Exception as e:
        print(f"[Worker] Could not set VRAM limit: {e}")

class MediumVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 3
        for _ in range(4): 
            layers.append(nn.Conv2d(in_channels, 256, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.ReLU(inplace=True))
            in_channels = 256
            
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1000)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class YOLOWorker:
    def __init__(self, job_config: dict, checkpoint_dir: str, coordinator_url: str, job_id: str):
        self.job_config = job_config
        self.checkpoint_dir = checkpoint_dir
        self.coordinator_url = coordinator_url
        self.job_id = job_id
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MediumVisionModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def run(self):
        print(f"[Worker] Starting Medium Workload for job {self.job_id} on {self.device}")
        total_epochs = self.job_config.get('epochs', 10)
        batch_size = self.job_config.get('batch_size', 32)
        print(f"[Worker] Using Batch Size: {batch_size}")
        
        start_epoch = 1
        existing_checkpoints = []
        if os.path.exists(self.checkpoint_dir):
            for f in os.listdir(self.checkpoint_dir):
                if f.endswith('.pt') and 'epoch_' in f:
                    try:
                        ep = int(f.split('epoch_')[1].split('.pt')[0])
                        existing_checkpoints.append((ep, os.path.join(self.checkpoint_dir, f)))
                    except: pass
                        
        if existing_checkpoints:
            existing_checkpoints.sort(key=lambda x: x[0])
            latest_ep, latest_path = existing_checkpoints[-1]
            try:
                checkpoint = torch.load(latest_path, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = latest_ep + 1
                print(f"[Worker] Resumed successfully from epoch {latest_ep}")
            except Exception as e:
                print(f"[Worker] Failed to load checkpoint: {e}")

        dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
        dummy_target = torch.randint(0, 1000, (batch_size,)).to(self.device)

        for epoch in range(start_epoch, total_epochs + 1):
            try:
                for step in range(30):
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
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[Worker] OUT OF MEMORY ERROR! GPU attempted to exceed the 5 GB hard limit.")
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                else: print(f"[Worker] RuntimeError: {e}")
                break
            except Exception as e:
                print(f"[Worker] Error during training: {e}")
                break

        final_path = os.path.join(self.checkpoint_dir, "final_model.pt")
        torch.save(self.model.state_dict(), final_path)
        
        return {
            "output_path": final_path,
            "output_hash": self.get_output_hash(final_path),
            "status": "completed"
        }

    def _save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, path)
        try:
            requests.post(f"{self.coordinator_url}/api/jobs/{self.job_id}/checkpoint", json={
                "epoch": epoch, "file_path": path, "file_hash": self.get_output_hash(path), "size_bytes": os.path.getsize(path)
            }, timeout=5)
        except: pass

    def _report_progress(self, epoch, total):
        try:
            requests.post(f"{self.coordinator_url}/api/jobs/{self.job_id}/progress", json={
                "epoch": epoch, "total_epochs": total, "percent": int((epoch / total) * 100), "eta_seconds": (total - epoch) * 5
            }, timeout=5)
        except: pass

    def get_output_hash(self, file_path):
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f: hasher.update(f.read())
            return hasher.hexdigest()
        except: return "hash_error"
