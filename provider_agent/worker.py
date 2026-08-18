import os
import json
import hashlib
import tempfile
import requests
import yaml
import sys
from pathlib import Path


def _verify_cuda_or_raise():
    """
    Check that PyTorch can see a CUDA GPU before we accept any job.
    Prints a clear diagnostic and raises RuntimeError if not available.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "[GPU CHECK FAILED] PyTorch is not installed.\n"
            "Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        )

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


class YOLOWorker:
    def __init__(self, job_config: dict, checkpoint_dir: str, coordinator_url: str, job_id: str):
        self.job_config = job_config
        self.checkpoint_dir = checkpoint_dir
        self.coordinator_url = coordinator_url
        self.job_id = job_id

        # Validate GPU FIRST — fail early, don't waste time loading the model
        self.device = _verify_cuda_or_raise()

        # Ensure checkpoint dir exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def run(self):
        try:
            from ultralytics import YOLO
            import ultralytics.nn.tasks as _ult_tasks
            import torch.serialization as _ts
        except ImportError:
            raise RuntimeError(
                "ultralytics package not found.\n"
                "Run: pip install ultralytics"
            )

        # ── PyTorch 2.6+ torch.load Monkeypatch ────────────────────────────
        # PyTorch 2.6 changed torch.load() to default weights_only=True,
        # which breaks ultralytics checkpoint loading. Monkeypatch torch.load
        # to force weights_only=False, bypassing the restrictions completely.
        import torch as _torch
        _original_load = _torch.load
        def _patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)
        _torch.load = _patched_load

        model_name = self.job_config.get('model', 'yolov8n.pt')

        # ── Stale model file guard ──────────────────────────────────────────
        # If the cached .pt file exists but fails to load (old format),
        # delete it so ultralytics re-downloads a fresh copy.
        import os as _os
        local_pt = _os.path.join(_os.path.dirname(__file__), model_name)
        if _os.path.exists(local_pt):
            try:
                import torch as _torch
                _torch.load(local_pt, map_location='cpu', weights_only=False)
            except Exception:
                print(f"[YOLOWorker] Stale/corrupt model file detected at {local_pt} — deleting for re-download.")
                _os.remove(local_pt)

        print(f"[YOLOWorker] Loading model: {model_name}")
        model = YOLO(model_name)

        dataset = self.job_config.get('dataset')
        if not dataset:
            print("[YOLOWorker] No dataset specified — creating synthetic demo dataset.")
            dataset = self._create_synthetic_dataset()

        epochs = int(self.job_config.get('epochs', 10))
        print(f"[YOLOWorker] Training {epochs} epoch(s) on GPU:{self.device}  dataset: {dataset}")

        try:
            for epoch in range(1, epochs + 1):
                print(f"[YOLOWorker] ── Epoch {epoch}/{epochs} ──")
                model.train(
                    data=dataset,
                    epochs=1,
                    save=True,
                    project=self.checkpoint_dir,
                    name=f"run_epoch_{epoch}",
                    exist_ok=True,
                    device=self.device,   # ← explicit GPU
                    verbose=False,        # suppress ultralytics spam
                )

                # Save & report checkpoint
                model_path = os.path.join(
                    self.checkpoint_dir, f"run_epoch_{epoch}", "weights", "last.pt"
                )
                if os.path.exists(model_path):
                    self._save_checkpoint(epoch, model_path)

                self._report_progress(epoch, epochs)

        except KeyboardInterrupt:
            print("[YOLOWorker] Training interrupted — saving checkpoint…")
            final_path = os.path.join(self.checkpoint_dir, "interrupted.pt")
            model.save(final_path)
            self._save_checkpoint("interrupted", final_path)
            raise

        # Locate best output
        final_model_path = os.path.join(
            self.checkpoint_dir, f"run_epoch_{epochs}", "weights", "best.pt"
        )
        if not os.path.exists(final_model_path):
            final_model_path = os.path.join(
                self.checkpoint_dir, f"run_epoch_{epochs}", "weights", "last.pt"
            )

        output_hash = (
            self.get_output_hash(final_model_path)
            if os.path.exists(final_model_path)
            else "unknown"
        )
        
        # Zip the output directory
        import shutil
        zip_path = os.path.join(self.checkpoint_dir, f"{self.job_id}_output")
        shutil.make_archive(zip_path, 'zip', self.checkpoint_dir)
        zip_file = f"{zip_path}.zip"

        return {
            "output_path": final_model_path,
            "output_hash": output_hash,
            "epochs_completed": epochs,
            "zip_file": zip_file
        }

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _create_synthetic_dataset(self) -> str:
        temp_dir = os.path.join(
            tempfile.gettempdir(), f"synthetic_dataset_{self.job_id}"
        )
        for split in ("train", "val"):
            os.makedirs(os.path.join(temp_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "labels", split), exist_ok=True)

        from PIL import Image

        for i in range(10):
            img = Image.new("RGB", (640, 640), color=(73, 109, 137))
            for split in ("train", "val"):
                img.save(os.path.join(temp_dir, "images", split, f"img_{i}.jpg"))
                with open(
                    os.path.join(temp_dir, "labels", split, f"img_{i}.txt"), "w"
                ) as f:
                    f.write("0 0.5 0.5 0.1 0.1\n")

        yaml_path = os.path.join(temp_dir, "dataset.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(
                {
                    "path": temp_dir,
                    "train": "images/train",
                    "val": "images/val",
                    "names": {0: "synthetic_object"},
                },
                f,
            )
        return yaml_path

    def _save_checkpoint(self, epoch, model_path):
        if not os.path.exists(model_path):
            return
        payload = {
            "epoch": epoch,
            "file_path": model_path,
            "file_hash": self.get_output_hash(model_path),
            "size_bytes": os.path.getsize(model_path),
        }
        try:
            requests.post(
                f"{self.coordinator_url}/api/jobs/{self.job_id}/checkpoint",
                json=payload,
                timeout=5,
            )
        except Exception as e:
            print(f"[YOLOWorker] Checkpoint report failed: {e}")

    def _report_progress(self, epoch, total):
        percent = (epoch / total) * 100.0 if total > 0 else 0
        payload = {
            "epoch": epoch,
            "total_epochs": total,
            "percent": percent,
            "eta_seconds": (total - epoch) * 60,
        }
        try:
            requests.post(
                f"{self.coordinator_url}/api/jobs/{self.job_id}/progress",
                json=payload,
                timeout=5,
            )
        except Exception as e:
            print(f"[YOLOWorker] Progress report failed: {e}")

    def get_output_hash(self, file_path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def resume_from_checkpoint(self, checkpoint_path: str) -> int:
        print(f"[YOLOWorker] Resuming from: {checkpoint_path}")
        return 1
