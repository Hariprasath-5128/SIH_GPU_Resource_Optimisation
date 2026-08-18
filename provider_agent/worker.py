import os
import json
import hashlib
import tempfile
import requests
import yaml
from pathlib import Path

class YOLOWorker:
    def __init__(self, job_config: dict, checkpoint_dir: str, coordinator_url: str, job_id: str):
        self.job_config = job_config
        self.checkpoint_dir = checkpoint_dir
        self.coordinator_url = coordinator_url
        self.job_id = job_id
        
        # Ensure checkpoint dir exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def run(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("ultralytics package not found. Please install it.")

        model_name = self.job_config.get('model', 'yolov8n.pt')
        print(f"[YOLOWorker] Loading model {model_name}")
        model = YOLO(model_name)

        dataset = self.job_config.get('dataset')
        if not dataset:
            print("[YOLOWorker] No dataset specified. Creating a synthetic demo dataset.")
            dataset = self._create_synthetic_dataset()

        epochs = self.job_config.get('epochs', 10)
        print(f"[YOLOWorker] Starting training for {epochs} epochs on dataset {dataset}")

        try:
            # We train for 1 epoch at a time in a loop to simulate the chunking/checkpoints
            for epoch in range(1, epochs + 1):
                # We can't really train just 1 epoch at a time iteratively without reloading easily in simple YOLO api,
                # but we will mimic it or just let it train and use a callback if we want real deep integration.
                # For simplicity here, let's just train 1 epoch iteratively.
                model.train(data=dataset, epochs=1, save=True, project=self.checkpoint_dir, name=f"run_epoch_{epoch}", exist_ok=True)
                
                # Report and save checkpoint
                model_path = os.path.join(self.checkpoint_dir, f"run_epoch_{epoch}", "weights", "last.pt")
                if os.path.exists(model_path):
                    self._save_checkpoint(epoch, model_path)
                
                self._report_progress(epoch, epochs)
        except KeyboardInterrupt:
            print("[YOLOWorker] Training interrupted! Saving final checkpoint...")
            final_path = os.path.join(self.checkpoint_dir, "interrupted.pt")
            model.save(final_path)
            self._save_checkpoint("interrupted", final_path)
            raise
        
        # Final output
        final_model_path = os.path.join(self.checkpoint_dir, f"run_epoch_{epochs}", "weights", "best.pt")
        if not os.path.exists(final_model_path):
            final_model_path = os.path.join(self.checkpoint_dir, f"run_epoch_{epochs}", "weights", "last.pt")
        
        output_hash = self.get_output_hash(final_model_path) if os.path.exists(final_model_path) else "unknown"
        
        return {
            "output_path": final_model_path,
            "output_hash": output_hash,
            "epochs_completed": epochs
        }

    def _create_synthetic_dataset(self) -> str:
        temp_dir = os.path.join(tempfile.gettempdir(), f"synthetic_dataset_{self.job_id}")
        os.makedirs(os.path.join(temp_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'labels', 'val'), exist_ok=True)
        
        # Create some blank images and empty labels
        from PIL import Image
        for i in range(10):
            img = Image.new('RGB', (640, 640), color = (73, 109, 137))
            img.save(os.path.join(temp_dir, 'images', 'train', f'img_{i}.jpg'))
            img.save(os.path.join(temp_dir, 'images', 'val', f'img_{i}.jpg'))
            with open(os.path.join(temp_dir, 'labels', 'train', f'img_{i}.txt'), 'w') as f:
                f.write("0 0.5 0.5 0.1 0.1\n")
            with open(os.path.join(temp_dir, 'labels', 'val', f'img_{i}.txt'), 'w') as f:
                f.write("0 0.5 0.5 0.1 0.1\n")
                
        yaml_content = {
            'path': temp_dir,
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'synthetic_object'}
        }
        yaml_path = os.path.join(temp_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
            
        return yaml_path

    def _save_checkpoint(self, epoch, model_path):
        if not os.path.exists(model_path):
            return
            
        size_bytes = os.path.getsize(model_path)
        file_hash = self.get_output_hash(model_path)
        
        url = f"{self.coordinator_url}/api/jobs/{self.job_id}/checkpoint"
        payload = {
            "epoch": epoch,
            "file_path": model_path,
            "file_hash": file_hash,
            "size_bytes": size_bytes
        }
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            print(f"[YOLOWorker] Failed to report checkpoint: {e}")

    def _report_progress(self, epoch, total):
        percent = (epoch / total) * 100.0 if total > 0 else 0
        eta_seconds = (total - epoch) * 60 # fake ETA
        url = f"{self.coordinator_url}/api/jobs/{self.job_id}/progress"
        payload = {
            "epoch": epoch,
            "total_epochs": total,
            "percent": percent,
            "eta_seconds": eta_seconds
        }
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            print(f"[YOLOWorker] Failed to report progress: {e}")

    def get_output_hash(self, file_path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def resume_from_checkpoint(self, checkpoint_path: str) -> int:
        print(f"[YOLOWorker] Resuming from checkpoint is not fully implemented in synthetic loop, but pretending to load: {checkpoint_path}")
        return 1
