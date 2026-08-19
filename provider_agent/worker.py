import time
import os
import hashlib
import requests
import torch
import torch.nn as nn
import torch.optim as optim

class HeavyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A deep, VRAM-hungry convolutional network to simulate heavy training
        layers = []
        in_channels = 3
        for _ in range(8): # 8 heavy convolutional layers
            layers.append(nn.Conv2d(in_channels, 512, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(512))
            layers.append(nn.ReLU(inplace=True))
            in_channels = 512
            
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1000)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class YOLOWorker:
    """
    Simulates a heavy AI vision training loop.
    """
    def __init__(self, job_config: dict, checkpoint_dir: str, coordinator_url: str, job_id: str):
        self.job_config = job_config
        self.checkpoint_dir = checkpoint_dir
        self.coordinator_url = coordinator_url
        self.job_id = job_id
        
        # Automatically use the NVIDIA GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HeavyVisionModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def run(self):
        print(f"[Worker] Starting Heavy Workload for job {self.job_id} on {self.device}")
        
        total_epochs = self.job_config.get('epochs', 10)
        
        # Dynamically read the batch size from the Consumer's request (Default: 32)
        batch_size = self.job_config.get('batch_size', 32)
        print(f"[Worker] Using Batch Size: {batch_size}")
        
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
                checkpoint = torch.load(latest_path, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = latest_ep + 1
                print(f"[Worker] Resumed successfully from epoch {latest_ep}")
            except Exception as e:
                print(f"[Worker] Failed to load checkpoint: {e}")

        # 2. Dynamic Dummy data pushing into the GPU based on batch size
        # Format: (Batch Size, Channels, Height, Width)
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
        dummy_target = torch.randint(0, 1000, (batch_size,)).to(self.device)

        # 3. The main AI training loop
        for epoch in range(start_epoch, total_epochs + 1):
            try:
                # Simulate heavy compute step to ramp up GPU utilization and Temp
                for step in range(30):
                    self.optimizer.zero_grad()
                    output = self.model(dummy_input)
                    loss = self.criterion(output, dummy_target)
                    loss.backward()
                    self.optimizer.step()
                    
                    # Small sleep so it doesn't instantly finish, letting the GPU heat up
                    time.sleep(0.05) 
                
                print(f"[Worker] Epoch {epoch}/{total_epochs} completed. Loss: {loss.item():.4f}")
                self._save_checkpoint(epoch)
                self._report_progress(epoch, total_epochs)
                
            except KeyboardInterrupt:
                print("[Worker] Interrupted by user. Saving emergency checkpoint...")
                self._save_checkpoint(epoch)
                break
            except RuntimeError as e:
                # Catch "Out Of Memory" errors gracefully so the agent doesn't explode
                if "out of memory" in str(e).lower():
                    print(f"[Worker] OUT OF MEMORY ERROR! Batch size {batch_size} is too large for this GPU's VRAM.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    print(f"[Worker] RuntimeError: {e}")
                break
            except Exception as e:
                print(f"[Worker] Error during training: {e}")
                break

        # 4. Final output generation
        final_path = os.path.join(self.checkpoint_dir, "final_model.pt")
        torch.save(self.model.state_dict(), final_path)
        
        return {
            "output_path": final_path,
            "output_hash": self.get_output_hash(final_path),
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