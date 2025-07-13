#!/usr/bin/env python3
"""Web-интерфейс для GPU докинга с использованием granian @Web

Этот модуль предоставляет веб-интерфейс для выполнения GPU-ускоренного молекулярного докинга
с использованием AutoDock-GPU и других инструментов.
"""

import json

# Добавляем путь к корневой директории проекта
import sys
import time
from pathlib import Path
from typing import Any

from granian import Granian
from granian.constants import Interfaces
from granian.rsgi import RSGIReceive, RSGIScope, RSGISend

sys.path.append(str(Path(__file__).parent))

from config import DOCKING_PARAMETERS
from step_04_hit_selection.accelerated_docking import AcceleratedDocking, HierarchicalDocking
from step_04_hit_selection.run_hit_selection import GPUAcceleratedDocking, optimize_docking_performance
from utils.logger import LOGGER as logger

# Глобальные переменные для хранения состояния
docking_engine = None
current_job_id = None
job_results = {}
job_status = {}


class GPUDockingAPI:
    """API для GPU докинга"""

    def __init__(self):
        self.docking_config = DOCKING_PARAMETERS.copy()
        self.initialize_docking_engine()

    def initialize_docking_engine(self):
        """Инициализация GPU докинг движка"""
        global docking_engine
        try:
            # Оптимизируем конфигурацию для текущей системы
            optimized_config = optimize_docking_performance()
            self.docking_config.update(optimized_config)

            # Создаем движок в зависимости от доступности GPU
            if self.docking_config.get("use_gpu", False):
                gpu_engine = self.docking_config.get("gpu_engine", "autodock_gpu")
                if gpu_engine == "autodock_gpu":
                    docking_engine = AcceleratedDocking(self.docking_config)
                    logger.info("Инициализирован AcceleratedDocking с AutoDock-GPU")
                else:
                    docking_engine = GPUAcceleratedDocking(self.docking_config)
                    logger.info("Инициализирован GPUAcceleratedDocking")
            else:
                docking_engine = AcceleratedDocking(self.docking_config)
                logger.info("Инициализирован AcceleratedDocking в CPU режиме")

        except Exception as e:
            logger.error(f"Ошибка инициализации GPU докинг движка: {e}")
            # Fallback к базовому CPU докингу
            docking_engine = AcceleratedDocking({"use_gpu": False})

    def get_status(self) -> dict[str, Any]:
        """Получить статус системы и доступность GPU"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_free": gpu.memoryFree,
                    "memory_used": gpu.memoryUsed,
                    "utilization": gpu.load * 100
                })
        except ImportError:
            gpu_info = []

        return {
            "gpu_available": self.docking_config.get("use_gpu", False),
            "gpu_engine": self.docking_config.get("gpu_engine", "vina_optimized"),
            "gpu_devices": gpu_info,
            "config": self.docking_config,
            "current_job": current_job_id,
            "job_count": len(job_results)
        }

    def start_docking(self, molecules: list[dict[str, Any]], job_id: str) -> dict[str, Any]:
        """Запустить докинг задачу"""
        global current_job_id, job_status

        try:
            if current_job_id is not None:
                return {
                    "success": False,
                    "error": f"Другая задача уже выполняется: {current_job_id}"
                }

            current_job_id = job_id
            job_status[job_id] = {
                "status": "running",
                "start_time": time.time(),
                "molecules_count": len(molecules),
                "completed_count": 0
            }

            # Запускаем докинг в отдельном потоке/процессе
            # Для упрощения выполняем синхронно
            logger.info(f"Запуск GPU докинга для {len(molecules)} молекул")

            if self.docking_config.get("use_hierarchical", False):
                hierarchical_docking = HierarchicalDocking(self.docking_config)
                results = hierarchical_docking.dock_molecules(molecules)
            else:
                results = docking_engine.dock_molecules_batch(molecules)

            # Сохраняем результаты
            job_results[job_id] = results
            job_status[job_id]["status"] = "completed"
            job_status[job_id]["end_time"] = time.time()
            job_status[job_id]["completed_count"] = len(results)

            current_job_id = None

            return {
                "success": True,
                "job_id": job_id,
                "results_count": len(results),
                "message": f"Докинг завершен успешно для {len(results)} молекул"
            }

        except Exception as e:
            logger.error(f"Ошибка при запуске докинга: {e}")
            job_status[job_id] = {
                "status": "failed",
                "error": str(e),
                "start_time": time.time(),
                "end_time": time.time()
            }
            current_job_id = None
            return {
                "success": False,
                "error": str(e)
            }

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Получить статус задачи"""
        if job_id not in job_status:
            return {"success": False, "error": "Задача не найдена"}

        status = job_status[job_id].copy()
        if job_id in job_results:
            status["results"] = job_results[job_id]

        return {"success": True, "status": status}

    def get_all_jobs(self) -> dict[str, Any]:
        """Получить все задачи"""
        return {
            "success": True,
            "jobs": job_status,
            "current_job": current_job_id
        }


# Создаем API экземпляр
api = GPUDockingAPI()


async def app(scope: RSGIScope, receive: RSGIReceive, send: RSGISend) -> None:
    """Главная RSGI приложение"""
    if scope["type"] == "http":
        path = scope["path"]
        method = scope["method"]

        # Простая маршрутизация
        if path == "/" and method == "GET":
            await handle_index(scope, receive, send)
        elif path == "/api/status" and method == "GET":
            await handle_status(scope, receive, send)
        elif path == "/api/dock" and method == "POST":
            await handle_dock(scope, receive, send)
        elif path.startswith("/api/job/") and method == "GET":
            job_id = path.split("/")[-1]
            await handle_job_status(scope, receive, send, job_id)
        elif path == "/api/jobs" and method == "GET":
            await handle_all_jobs(scope, receive, send)
        else:
            await handle_404(scope, receive, send)


async def handle_index(scope: RSGIScope, receive: RSGIReceive, send: RSGISend) -> None:
    """Обработка главной страницы"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GPU Молекулярный Докинг</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .status { background: #f0f0f0; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .gpu-info { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .error { background: #ffe6e6; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .success { background: #e6ffe6; padding: 15px; border-radius: 5px; margin: 10px 0; }
            button { padding: 10px 20px; margin: 10px; cursor: pointer; }
            textarea { width: 100%; height: 200px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>GPU Молекулярный Докинг</h1>
            <div id="status" class="status">Загрузка статуса...</div>
            
            <h2>Запуск Докинга</h2>
            <div>
                <label>Молекулы (JSON):</label>
                <textarea id="molecules" placeholder='[{"id": "mol1", "smiles": "CCO"}, {"id": "mol2", "smiles": "CCC"}]'></textarea>
                <button onclick="startDocking()">Запустить Докинг</button>
            </div>
            
            <div id="results"></div>
            
            <h2>Задачи</h2>
            <div id="jobs"></div>
        </div>
        
        <script>
            async function loadStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    document.getElementById('status').innerHTML = `
                        <h3>Статус Системы</h3>
                        <p>GPU доступен: ${data.gpu_available ? 'Да' : 'Нет'}</p>
                        <p>GPU движок: ${data.gpu_engine}</p>
                        <p>Текущая задача: ${data.current_job || 'Нет'}</p>
                        <p>Количество задач: ${data.job_count}</p>
                        ${data.gpu_devices.length > 0 ? '<h4>GPU Устройства:</h4>' + data.gpu_devices.map(gpu => `
                            <div class="gpu-info">
                                <strong>${gpu.name}</strong><br>
                                Память: ${gpu.memory_used}MB / ${gpu.memory_total}MB<br>
                                Использование: ${gpu.utilization.toFixed(1)}%
                            </div>
                        `).join('') : ''}
                    `;
                } catch (error) {
                    document.getElementById('status').innerHTML = `<div class="error">Ошибка загрузки статуса: ${error.message}</div>`;
                }
            }
            
            async function startDocking() {
                const molecules = document.getElementById('molecules').value;
                const jobId = 'job_' + Date.now();
                
                try {
                    const moleculesData = JSON.parse(molecules);
                    const response = await fetch('/api/dock', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            molecules: moleculesData,
                            job_id: jobId
                        })
                    });
                    
                    const data = await response.json();
                    const resultDiv = document.getElementById('results');
                    
                    if (data.success) {
                        resultDiv.innerHTML = `<div class="success">Докинг запущен: ${data.message}</div>`;
                        setTimeout(() => loadJobs(), 1000);
                    } else {
                        resultDiv.innerHTML = `<div class="error">Ошибка: ${data.error}</div>`;
                    }
                } catch (error) {
                    document.getElementById('results').innerHTML = `<div class="error">Ошибка: ${error.message}</div>`;
                }
            }
            
            async function loadJobs() {
                try {
                    const response = await fetch('/api/jobs');
                    const data = await response.json();
                    const jobsDiv = document.getElementById('jobs');
                    
                    if (data.success) {
                        jobsDiv.innerHTML = Object.entries(data.jobs).map(([jobId, job]) => `
                            <div class="job">
                                <h4>${jobId}</h4>
                                <p>Статус: ${job.status}</p>
                                <p>Молекул: ${job.molecules_count || 'N/A'}</p>
                                <p>Завершено: ${job.completed_count || 0}</p>
                                ${job.error ? `<p class="error">Ошибка: ${job.error}</p>` : ''}
                            </div>
                        `).join('');
                    }
                } catch (error) {
                    document.getElementById('jobs').innerHTML = `<div class="error">Ошибка загрузки задач: ${error.message}</div>`;
                }
            }
            
            // Инициализация
            loadStatus();
            loadJobs();
            
            // Обновление каждые 5 секунд
            setInterval(() => {
                loadStatus();
                loadJobs();
            }, 5000);
        </script>
        </body>
    </html>
    """

    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [[b"content-type", b"text/html; charset=utf-8"]]
    })
    await send({
        "type": "http.response.body",
        "body": html_content.encode("utf-8")
    })


async def handle_status(scope: RSGIScope, receive: RSGIReceive, send: RSGISend) -> None:
    """Обработка запроса статуса"""
    status = api.get_status()
    response_data = json.dumps(status, indent=2)

    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [[b"content-type", b"application/json"]]
    })
    await send({
        "type": "http.response.body",
        "body": response_data.encode("utf-8")
    })


async def handle_dock(scope: RSGIScope, receive: RSGIReceive, send: RSGISend) -> None:
    """Обработка запроса на докинг"""
    # Получаем данные из тела запроса
    body = b""
    async for message in receive():
        if message["type"] == "http.request":
            body += message.get("body", b"")
        elif message["type"] == "http.disconnect":
            break

    try:
        request_data = json.loads(body.decode("utf-8"))
        molecules = request_data.get("molecules", [])
        job_id = request_data.get("job_id", f"job_{int(time.time())}")

        result = api.start_docking(molecules, job_id)
        response_data = json.dumps(result, indent=2)

        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [[b"content-type", b"application/json"]]
        })
        await send({
            "type": "http.response.body",
            "body": response_data.encode("utf-8")
        })

    except Exception as e:
        error_response = json.dumps({"success": False, "error": str(e)})
        await send({
            "type": "http.response.start",
            "status": 400,
            "headers": [[b"content-type", b"application/json"]]
        })
        await send({
            "type": "http.response.body",
            "body": error_response.encode("utf-8")
        })


async def handle_job_status(scope: RSGIScope, receive: RSGIReceive, send: RSGISend, job_id: str) -> None:
    """Обработка запроса статуса задачи"""
    result = api.get_job_status(job_id)
    response_data = json.dumps(result, indent=2)

    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [[b"content-type", b"application/json"]]
    })
    await send({
        "type": "http.response.body",
        "body": response_data.encode("utf-8")
    })


async def handle_all_jobs(scope: RSGIScope, receive: RSGIReceive, send: RSGISend) -> None:
    """Обработка запроса всех задач"""
    result = api.get_all_jobs()
    response_data = json.dumps(result, indent=2)

    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [[b"content-type", b"application/json"]]
    })
    await send({
        "type": "http.response.body",
        "body": response_data.encode("utf-8")
    })


async def handle_404(scope: RSGIScope, receive: RSGIReceive, send: RSGISend) -> None:
    """Обработка 404 ошибки"""
    await send({
        "type": "http.response.start",
        "status": 404,
        "headers": [[b"content-type", b"text/plain"]]
    })
    await send({
        "type": "http.response.body",
        "body": b"Not Found"
    })


def main():
    """Запуск веб-сервера"""
    logger.info("Запуск GPU Docking Web Server...")

    # Создаем и запускаем сервер
    server = Granian(
        "web_gpu_docking:app",
        interface=Interfaces.RSGI,
        host="0.0.0.0",
        port=8000,
        workers=1,
        reload=True
    )

    logger.info("Сервер запущен на http://0.0.0.0:8000")
    server.serve()


if __name__ == "__main__":
    main()
