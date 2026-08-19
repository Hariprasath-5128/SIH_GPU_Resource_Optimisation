import time
import os
import hashlib
import requests
import torch
import torch.nn as nn
import torch.optim as optim



class SmallVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Lightweight 2-conv model — fast epochs, tiny .pt, light on VRAM
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class YOLOWorker:
    def __init__(self, job_config: dict, checkpoint_dir: str, coordinator_url: str, job_id: str):
        self.job_config = job_config
        self.checkpoint_dir = checkpoint_dir
        self.coordinator_url = coordinator_url
        self.job_id = job_id
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SmallVisionModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def run(self):
        print(f"[Worker] Starting Small Workload for job {self.job_id} on {self.device}")
        total_epochs = self.job_config.get('epochs', 10)
        batch_size = self.job_config.get('batch_size', 16)
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

        # Small dummy tensors — light on VRAM, fast forward pass
        dummy_input = torch.randn(batch_size, 3, 32, 32).to(self.device)
        dummy_target = torch.randint(0, 10, (batch_size,)).to(self.device)

        # Modest matmul to give a small but visible GPU utilisation blip
        compute_A = torch.randn(2048, 2048, device=self.device)
        compute_B = torch.randn(2048, 2048, device=self.device)

        for epoch in range(start_epoch, total_epochs + 1):
            try:
                for step in range(10):  # short loop — stays light
                    _ = torch.matmul(compute_A, compute_B)
                    self.optimizer.zero_grad()
                    output = self.model(dummy_input)
                    loss = self.criterion(output, dummy_target)
                    loss.backward()
                    self.optimizer.step()
                    time.sleep(0.02)
                
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
        
        # Explicitly flush everything so VRAM drops immediately back to zero after transfer
        print("[Worker] Flushing GPU memory buffer...")
        try:
            del dummy_input
            del dummy_target
            del compute_A
            del compute_B
            del self.model
            del self.optimizer
        except Exception:
            pass
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "output_path": final_path,
            "output_hash": self.get_output_hash(final_path),
            "status": "completed"
        }

    def _save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        # Save only model_state_dict (not optimizer) so each checkpoint is ~85MB, not ~255MB
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict()}, path)
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
