import requests
import time
import sys

COORD_URL = "http://localhost:8000"

def run_test():
    print("=== Submitting test job to Coordinator ===")
    payload = {
        "consumer_id": "tester_01",
        "workload_type": "YOLO Training",
        "model_name": "yolov8n.pt",
        "vram_required_gb": 2.0,
        "max_budget": 50.0,
        "estimated_minutes": 10,
        "input_hash": None
    }
    
    try:
        res = requests.post(f"{COORD_URL}/api/jobs/submit", json=payload)
        res.raise_for_status()
        job = res.json()
        job_id = job.get("job_id")
        provider_node = job.get("provider_node")
        print(f"Job submitted successfully! Job ID: {job_id}")
        print(f"Matched Provider Node: {provider_node}")
    except Exception as e:
        print(f"Failed to submit job: {e}")
        return

    print("\nPolling job status...")
    for i in range(30):
        try:
            r = requests.get(f"{COORD_URL}/api/jobs")
            jobs = r.json()
            if isinstance(jobs, dict):
                jobs = jobs.get("jobs", [])
            my_job = next((j for j in jobs if j["job_id"] == job_id), None)
            
            if my_job:
                status = my_job["status"]
                progress = my_job.get("progress", 0.0)
                
                # Use carriage return to update the same line
                sys.stdout.write(f"\rStatus: {status:10} | Progress: {progress:.1f}%")
                sys.stdout.flush()
                
                if status == "COMPLETED":
                    print(f"\n[SUCCESS] Job completed. Output Hash: {my_job.get('output_hash')}")
                    return
                elif status in ["FAILED", "INTERRUPTED"]:
                    print(f"\n[FAILED] Job ended with status: {status}")
                    return
            else:
                print("Job not found in queue.")
                
        except Exception as e:
            print(f"\nError checking status: {e}")
        
        time.sleep(2)
        
    print("\nTest timed out.")

if __name__ == "__main__":
    run_test()
