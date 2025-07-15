#!/usr/bin/env python3
"""GPU Monitoring Script for Docking Operations
Monitors GPU utilization, memory usage, and performance during AutoDock-GPU runs
"""

import json
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil


class GPUMonitor:
    """Monitors GPU performance during docking operations"""

    def __init__(self, log_file: str = "gpu_monitoring.log", interval: float = 1.0):
        self.log_file = Path(log_file)
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.stats = []

    def start_monitoring(self):
        """Start GPU monitoring in a separate thread"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"🔍 GPU мониторинг запущен, интервал: {self.interval}s")

    def stop_monitoring(self):
        """Stop GPU monitoring"""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        # Save final stats
        self._save_stats()
        print("⏹️  GPU мониторинг остановлен")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get GPU stats
                gpu_stats = self._get_gpu_stats()

                # Get CPU stats
                cpu_stats = self._get_cpu_stats()

                # Get memory stats
                memory_stats = self._get_memory_stats()

                # Get process stats
                process_stats = self._get_autodock_process_stats()

                # Combine all stats
                timestamp = datetime.now().isoformat()
                combined_stats = {
                    "timestamp": timestamp,
                    "gpu": gpu_stats,
                    "cpu": cpu_stats,
                    "memory": memory_stats,
                    "processes": process_stats,
                }

                self.stats.append(combined_stats)

                # Print real-time stats
                self._print_realtime_stats(combined_stats)

                # Save to log file periodically
                if len(self.stats) % 10 == 0:
                    self._save_stats()

            except Exception as e:
                print(f"❌ Ошибка мониторинга: {e}")

            time.sleep(self.interval)

    def _get_gpu_stats(self) -> dict:
        """Get GPU utilization and memory stats"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                return {
                    "utilization": float(values[0]),
                    "memory_used": float(values[1]),
                    "memory_total": float(values[2]),
                    "temperature": float(values[3]),
                    "power_draw": float(values[4]) if values[4] != "[N/A]" else 0.0,
                }
            return {"error": result.stderr}

        except Exception as e:
            return {"error": str(e)}

    def _get_cpu_stats(self) -> dict:
        """Get CPU utilization stats"""
        try:
            return {
                "utilization": psutil.cpu_percent(interval=None),
                "cores": psutil.cpu_count(),
                "load_avg": psutil.getloadavg(),
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_memory_stats(self) -> dict:
        """Get system memory stats"""
        try:
            memory = psutil.virtual_memory()
            return {
                "used_gb": memory.used / (1024**3),
                "total_gb": memory.total / (1024**3),
                "percent": memory.percent,
                "available_gb": memory.available / (1024**3),
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_autodock_process_stats(self) -> list:
        """Get AutoDock-GPU process statistics"""
        try:
            processes = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
                try:
                    if "autodock" in proc.info["name"].lower():
                        processes.append(
                            {
                                "pid": proc.info["pid"],
                                "name": proc.info["name"],
                                "cpu_percent": proc.info["cpu_percent"],
                                "memory_mb": proc.info["memory_info"].rss / (1024**2),
                            }
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return processes
        except Exception as e:
            return [{"error": str(e)}]

    def _print_realtime_stats(self, stats: dict):
        """Print real-time statistics"""
        timestamp = stats["timestamp"].split("T")[1][:8]

        gpu = stats["gpu"]
        cpu = stats["cpu"]
        memory = stats["memory"]
        processes = stats["processes"]

        if "error" not in gpu:
            gpu_util = gpu["utilization"]
            gpu_mem = gpu["memory_used"]
            gpu_mem_total = gpu["memory_total"]
            gpu_temp = gpu["temperature"]
            gpu_power = gpu["power_draw"]

            # Color coding for GPU utilization
            if gpu_util > 80:
                util_color = "🔥"
            elif gpu_util > 50:
                util_color = "🟡"
            elif gpu_util > 20:
                util_color = "🟢"
            else:
                util_color = "🔵"

            print(
                f"\r{timestamp} | GPU: {util_color}{gpu_util:5.1f}% | "
                f"VRAM: {gpu_mem:5.0f}/{gpu_mem_total:5.0f}MB | "
                f"CPU: {cpu['utilization']:5.1f}% | "
                f"RAM: {memory['used_gb']:5.1f}GB | "
                f"Temp: {gpu_temp:2.0f}°C | "
                f"Power: {gpu_power:3.0f}W | "
                f"Processes: {len(processes)}",
                end="",
            )
        else:
            print(f"\r{timestamp} | GPU: ERROR | CPU: {cpu['utilization']:5.1f}% | RAM: {memory['used_gb']:5.1f}GB", end="")

    def _save_stats(self):
        """Save statistics to log file"""
        try:
            with open(self.log_file, "w") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"❌ Ошибка сохранения статистики: {e}")

    def get_performance_summary(self) -> dict:
        """Get performance summary"""
        if not self.stats:
            return {"error": "No statistics available"}

        gpu_utils = [s["gpu"]["utilization"] for s in self.stats if "error" not in s["gpu"]]
        cpu_utils = [s["cpu"]["utilization"] for s in self.stats if "error" not in s["cpu"]]

        if not gpu_utils:
            return {"error": "No valid GPU statistics"}

        return {
            "duration_seconds": len(self.stats) * self.interval,
            "gpu_utilization": {"avg": sum(gpu_utils) / len(gpu_utils), "max": max(gpu_utils), "min": min(gpu_utils)},
            "cpu_utilization": {"avg": sum(cpu_utils) / len(cpu_utils), "max": max(cpu_utils), "min": min(cpu_utils)},
            "total_samples": len(self.stats),
        }


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Получен сигнал остановки...")
    sys.exit(0)


def main():
    """Main function for standalone GPU monitoring"""
    import argparse

    parser = argparse.ArgumentParser(description="GPU Performance Monitor for AutoDock-GPU")
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds")
    parser.add_argument("--log-file", default="gpu_monitoring.log", help="Log file path")
    parser.add_argument("--duration", type=int, help="Monitoring duration in seconds")

    args = parser.parse_args()

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Create monitor
    monitor = GPUMonitor(log_file=args.log_file, interval=args.interval)

    try:
        monitor.start_monitoring()

        if args.duration:
            time.sleep(args.duration)
            monitor.stop_monitoring()

            # Print summary
            summary = monitor.get_performance_summary()
            print("\n\n📊 Сводка производительности:")
            print(f"   Длительность: {summary['duration_seconds']:.1f}s")
            print(
                f"   GPU утилизация: {summary['gpu_utilization']['avg']:.1f}% (avg), {summary['gpu_utilization']['max']:.1f}% (max)"
            )
            print(
                f"   CPU утилизация: {summary['cpu_utilization']['avg']:.1f}% (avg), {summary['cpu_utilization']['max']:.1f}% (max)"
            )
        else:
            # Run indefinitely
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Остановка мониторинга...")
        monitor.stop_monitoring()

        # Print summary
        summary = monitor.get_performance_summary()
        if "error" not in summary:
            print("\n📊 Сводка производительности:")
            print(f"   Длительность: {summary['duration_seconds']:.1f}s")
            print(
                f"   GPU утилизация: {summary['gpu_utilization']['avg']:.1f}% (avg), {summary['gpu_utilization']['max']:.1f}% (max)"
            )
            print(
                f"   CPU утилизация: {summary['cpu_utilization']['avg']:.1f}% (avg), {summary['cpu_utilization']['max']:.1f}% (max)"
            )


if __name__ == "__main__":
    main()
