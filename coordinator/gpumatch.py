import socket
import time

GPU_BENCHMARK_TABLE = {
    "RTX 4090": 100, "RTX 4080": 95, "RTX 4080 SUPER": 96,
    "RTX 4070 Ti": 91, "RTX 4070 Ti SUPER": 92, "RTX 4070": 87,
    "RTX 4070 SUPER": 88, "RTX 4060 Ti": 82, "RTX 4060": 76,
    "RTX 3090": 90, "RTX 3080": 84, "RTX 3070": 78,
    "RTX 3060 Ti": 74, "RTX 3060": 70, "RTX 3050": 60,
}

CLOUD_GPU_RATE = 87.0  # cloud equivalent rate per hour in rupees

def ping_latency(ip: str, port: int) -> float:
    try:
        start_time = time.time()
        with socket.create_connection((ip, port), timeout=1.0):
            return (time.time() - start_time) * 1000
    except (socket.timeout, ConnectionRefusedError, socket.error):
        return 10.0  # default 10ms if unreachable or local

def predict_availability(metrics_history: list, estimated_minutes: int) -> float:
    if not metrics_history:
        return 0.9 # Default high availability if no history
    
    # Calculate rolling average of GPU utilization
    recent_metrics = metrics_history[-10:]
    avg_utilization = sum(m.get('gpu_utilization', 0.0) for m in recent_metrics) / len(recent_metrics)
    
    # High utilization means lower availability for new jobs
    availability = max(0.1, 1.0 - (avg_utilization / 100.0))
    
    # Factor in recent failures or downtime indirectly (assuming history gaps)
    # But here we just use utilization for simplicity.
    return float(availability)

def compute_expected_cost(nominal_cost: float, avail_prob: float) -> float:
    # Adds failure-restart penalty
    failure_prob = 1.0 - avail_prob
    restart_penalty = nominal_cost * failure_prob * 0.5 # Assume 50% extra cost on failure
    return nominal_cost + restart_penalty

def compute_gpumatch_score(node_dict: dict, job_dict: dict, latest_metrics_dict: dict = None) -> dict:
    if latest_metrics_dict is None:
        latest_metrics_dict = {}

    vram_required = job_dict.get("vram_required_gb", 0.0)
    node_vram = node_dict.get("vram_gb", 0.0)
    avail_vram = latest_metrics_dict.get("available_vram_gb", node_vram)
    
    eligible = avail_vram >= vram_required and node_dict.get("status") == "online"
    
    if not eligible:
        return {
            "eligible": False,
            "final_score": 0.0,
            "breakdown": {},
            "expected_cost": 0.0,
            "nominal_cost": 0.0,
            "cloud_equivalent_cost": 0.0,
            "savings_percent": 0.0
        }

    # Performance
    gpu_model = node_dict.get("gpu_model", "")
    base_score = GPU_BENCHMARK_TABLE.get(gpu_model, 50.0)
    perf_score = base_score / 100.0

    # Reliability (from trust score)
    trust_score = node_dict.get("trust_score", 80.0)
    reliability_score = trust_score / 100.0
    
    # Availability
    # We mock metrics history since we might not have it in this pure function without DB access
    avail_prob = 0.95 # Assumed baseline
    
    # Network
    ip = node_dict.get("ip_address", "127.0.0.1")
    port = node_dict.get("port", 8000)
    latency_ms = ping_latency(ip, port)
    net_score = max(0.0, 1.0 - (latency_ms / 100.0)) # Normalize assuming 100ms is terrible

    # Cost Efficiency
    est_mins = job_dict.get("estimated_minutes", 60)
    est_hours = est_mins / 60.0
    price_per_hour = node_dict.get("price_per_hour", 10.0)
    
    nominal_cost = est_hours * price_per_hour
    cloud_cost = est_hours * CLOUD_GPU_RATE
    
    savings = cloud_cost - nominal_cost
    savings_percent = (savings / cloud_cost * 100.0) if cloud_cost > 0 else 0.0
    cost_eff_score = min(1.0, max(0.0, savings_percent / 100.0))
    
    expected_cost = compute_expected_cost(nominal_cost, avail_prob)
    
    # Energy - Simplified to 0.8 placeholder
    energy_score = 0.8
    
    final_score = (
        0.30 * perf_score * 100 +
        0.20 * reliability_score * 100 +
        0.15 * avail_prob * 100 +
        0.15 * net_score * 100 +
        0.10 * (trust_score / 100.0) * 100 +
        0.10 * cost_eff_score * 100
    )

    return {
        "eligible": eligible,
        "final_score": round(final_score, 2),
        "breakdown": {
            "performance": round(perf_score * 100, 2),
            "reliability": round(reliability_score * 100, 2),
            "availability": round(avail_prob * 100, 2),
            "network": round(net_score * 100, 2),
            "trust": round(trust_score, 2),
            "cost_efficiency": round(cost_eff_score * 100, 2),
            "energy": round(energy_score * 100, 2)
        },
        "expected_cost": round(expected_cost, 2),
        "nominal_cost": round(nominal_cost, 2),
        "cloud_equivalent_cost": round(cloud_cost, 2),
        "savings_percent": round(savings_percent, 2)
    }
