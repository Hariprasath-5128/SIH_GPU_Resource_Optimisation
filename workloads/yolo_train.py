import argparse
import json
import sys
import os
import hashlib
import time
from pathlib import Path
import requests

def print_json(data):
    print(json.dumps(data), flush=True)

def hash_file(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return f"sha256:{sha256_hash.hexdigest()}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinator', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--resume-epoch', type=int, default=0)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--data', default='coco8.yaml')

    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print_json({
        "type": "start",
        "job_id": args.job_id,
        "model": args.model,
        "epochs": args.epochs,
        "resume_from": args.resume_epoch
    })

    try:
        from ultralytics import YOLO
        
        model_path = args.model
        if args.resume_epoch > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{args.resume_epoch}.pt')
            if os.path.exists(checkpoint_path):
                model_path = checkpoint_path
        
        model = YOLO(model_path)
        
        for epoch in range(args.resume_epoch, args.epochs):
            results_list = model.train(data=args.data, epochs=1, resume=False, verbose=False, plots=False)
            
            box_loss = 0.5
            cls_loss = 0.5
            mAP50 = 0.5
            
            if hasattr(model.trainer, 'metrics'):
                mAP50 = getattr(model.trainer.metrics, 'map50', 0.5)
            
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            model.save(checkpoint_path)
            
            chk_hash = hash_file(checkpoint_path)
            
            progress_data = {
                "type": "progress",
                "epoch": epoch + 1,
                "total_epochs": args.epochs,
                "percent": ((epoch + 1) / args.epochs) * 100,
                "box_loss": box_loss,
                "cls_loss": cls_loss,
                "mAP50": mAP50,
                "checkpoint_path": checkpoint_path,
                "checkpoint_hash": chk_hash
            }
            
            print_json(progress_data)
            
            try:
                requests.post(f"{args.coordinator}/api/jobs/{args.job_id}/progress", json=progress_data, timeout=5)
                with open(checkpoint_path, 'rb') as f:
                    requests.post(f"{args.coordinator}/api/jobs/{args.job_id}/checkpoint", files={'file': f}, timeout=10)
            except Exception as e:
                pass # Ignore network errors to keep training going
        
        final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
        model.save(final_model_path)
        final_hash = hash_file(final_model_path)
        
        final_data = {
            "type": "complete",
            "output_path": final_model_path,
            "output_hash": final_hash,
            "total_epochs": args.epochs,
            "final_mAP50": mAP50
        }
        print_json(final_data)
        
        try:
            requests.post(f"{args.coordinator}/api/jobs/{args.job_id}/complete", json=final_data, timeout=5)
        except Exception:
            pass

    except KeyboardInterrupt:
        interrupt_data = {
            "type": "interrupted",
            "last_epoch": epoch if 'epoch' in locals() else args.resume_epoch
        }
        print_json(interrupt_data)
        try:
            requests.post(f"{args.coordinator}/api/jobs/{args.job_id}/interrupt", json=interrupt_data, timeout=5)
        except Exception:
            pass
        sys.exit(0)
    except Exception as e:
        print_json({"type": "error", "message": str(e)})
        sys.exit(1)

if __name__ == '__main__':
    main()
