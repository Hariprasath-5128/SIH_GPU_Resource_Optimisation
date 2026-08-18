import pynvml
import psutil
import time
from datetime import datetime

GPU_BENCHMARK_TABLE = {
    'RTX 4090': 200.0,
    'RTX 4080': 150.0,
    'RTX 4070': 100.0,
    'RTX 3090': 130.0,
    'RTX 3080': 90.0,
    'RTX 3070': 70.0,
    'T4': 50.0,
    'V100': 120.0,
    'A100': 300.0,
    'A10g': 140.0
}

class GPUProbe:
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            if self.device_count == 0:
                raise RuntimeError('No NVIDIA GPU found (device count 0)')
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._last_net_bytes = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
            self._last_net_time = time.time()
        except pynvml.NVMLError as e:
            raise RuntimeError(f'No NVIDIA GPU found or NVML error: {e}')

    def get_metrics(self) -> dict:
        try:
            gpu_model = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(gpu_model, bytes):
                gpu_model = gpu_model.decode('utf-8')
            
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            vram_gb = memory.total / (1024 ** 3)
            vram_used_gb = memory.used / (1024 ** 3)
            available_vram_gb = memory.free / (1024 ** 3)
            
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_utilization = float(utilization.gpu)
            
            temperature = float(pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU))
            
            try:
                power_watts = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            except pynvml.NVMLError:
                power_watts = 0.0
                
            try:
                power_limit_watts = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000.0
            except pynvml.NVMLError:
                power_limit_watts = 0.0
                
            cuda_version_raw = pynvml.nvmlSystemGetCudaDriverVersion()
            cuda_version = f"{cuda_version_raw // 1000}.{(cuda_version_raw % 100) // 10}"
            
            cpu_utilization = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            ram_used_gb = mem.used / (1024 ** 3)
            ram_total_gb = mem.total / (1024 ** 3)
            ram_available_gb = mem.available / (1024 ** 3)
            
            net_counters = psutil.net_io_counters()
            net_bytes = net_counters.bytes_sent + net_counters.bytes_recv
            current_time = time.time()
            time_diff = current_time - self._last_net_time
            if time_diff > 0:
                network_mbps = ((net_bytes - self._last_net_bytes) * 8) / (1024 * 1024 * time_diff)
            else:
                network_mbps = 100.0
                
            self._last_net_bytes = net_bytes
            self._last_net_time = current_time

            return {
                'gpu_model': gpu_model,
                'vram_gb': vram_gb,
                'available_vram_gb': available_vram_gb,
                'vram_used_gb': vram_used_gb,
                'gpu_utilization': gpu_utilization,
                'temperature': temperature,
                'power_watts': power_watts,
                'power_limit_watts': power_limit_watts,
                'cuda_version': cuda_version,
                'cpu_utilization': cpu_utilization,
                'ram_used_gb': ram_used_gb,
                'ram_total_gb': ram_total_gb,
                'ram_available_gb': ram_available_gb,
                'network_mbps': network_mbps,
                'timestamp': datetime.utcnow().isoformat()
            }
        except pynvml.NVMLError as e:
            print(f"Error fetching GPU metrics: {e}")
            return {
                'gpu_model': 'Unknown',
                'vram_gb': 0.0,
                'available_vram_gb': 0.0,
                'vram_used_gb': 0.0,
                'gpu_utilization': 0.0,
                'temperature': 0.0,
                'power_watts': 0.0,
                'power_limit_watts': 0.0,
                'cuda_version': '0.0',
                'cpu_utilization': psutil.cpu_percent(),
                'ram_used_gb': 0.0,
                'ram_total_gb': 0.0,
                'ram_available_gb': 0.0,
                'network_mbps': 100.0,
                'timestamp': datetime.utcnow().isoformat()
            }

    def get_gpu_model_clean(self) -> str:
        try:
            name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            for key in GPU_BENCHMARK_TABLE.keys():
                if key in name:
                    return key
            return name.replace('NVIDIA', '').replace('GeForce', '').strip()
        except pynvml.NVMLError:
            return 'Unknown GPU'

    def get_benchmark_score(self) -> float:
        clean_name = self.get_gpu_model_clean()
        for key, score in GPU_BENCHMARK_TABLE.items():
            if key in clean_name:
                return score
        return 70.0

    def close(self):
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
