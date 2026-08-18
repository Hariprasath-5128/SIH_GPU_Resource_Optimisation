# GPUShare — Intelligent One-to-One Peer GPU Sharing

## What it does
GPUShare is a peer-to-peer VS Code extension that enables secure, on-demand GPU resource sharing between trusted computers (e.g., between friends' laptops). It orchestrates complex machine learning workloads (like YOLO training) while optimizing resource allocation based on unique metrics like GPUMatch, predicted availability, and performance-per-watt. 

## Architecture
```
Consumer Laptop (VS Code) 
         |
   [Coordinator] <-----(REST API / WS)-----> Provider Laptop (RTX 4070)
         |                                           |
         |                                     [Provider Agent]
         |                                           |
         v                                           v
[Job Submission]                              [YOLO Worker] -> Checkpoint loop
```

## Quick Start

### Consumer Machine (Your Laptop)
1. Clone/copy project
2. Run `setup_consumer.bat`
3. Install VS Code extension: `code --install-extension extension/` or open project and press F5
4. Open GPUShare panel in VS Code Activity Bar
5. Note your IP address shown in coordinator startup output

### Provider Machine (Friend's Laptop)
1. Copy project or git clone
2. Run `setup_provider.bat`
3. Enter consumer's IP when prompted
4. Install VS Code extension (same steps)
5. Select Provider role in extension settings

## Requirements
- Python 3.10+
- NVIDIA GPU with CUDA (provider only)
- VS Code 1.85+
- Both machines on same WiFi network (or use ngrok for internet)

## Novel Features
1. **GPUMatch Score** — Multi-objective GPU selection (not just cheapest)
2. **Predictive Availability** — P(GPU stays free)
3. **Trust Score** — Provider reliability tracking
4. **Dynamic Pricing** — Demand-aware rates
5. **Expected Cost** — Includes failure risk in pricing
6. **Checkpoint Recovery** — Resume on disconnect
7. **Local-First Discovery** — LAN before internet
8. **Energy Scoring** — Performance per watt
9. **GPU Time-Slicing** — Resource limits
10. **Proof-of-Execution** — SHA-256 output verification

## Demo Scenario (for review)
1. **Job Submission:** Consumer selects YOLO workload and submits.
2. **GPUMatch Scoring:** Coordinator queries available Providers. Provider returns system stats, and Coordinator computes a GPUMatch score.
3. **Execution:** Highest score Provider is chosen. Provider Agent polls pending jobs, pulls the task, and spawns the YOLO Worker.
4. **Monitoring:** Worker streams real-time stdout metrics and checkpoint logs back to Consumer's VS Code via WebSocket.
5. **Interruption Recovery:** Consumer intentionally interrupts Provider. Worker saves state. Consumer resubmits and Worker resumes training from the latest checkpoint hash seamlessly.

## Technology Stack
| Component | Technology |
|---|---|
| VS Code Extension | JavaScript (Node.js) |
| Coordinator | Python FastAPI + SQLite |
| Provider Agent | Python + pynvml + psutil |
| GPU Workload | Ultralytics YOLO + PyTorch |
| Real-time | WebSocket |
| Communication | REST + WebSocket (LAN) |
