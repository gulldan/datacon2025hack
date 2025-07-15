#!/usr/bin/env python3
"""Простой Flask веб-сервер для демонстрации GPU докинга"""

import json
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

# Добавляем путь к корневой директории проекта
sys.path.append(str(Path(__file__).parent))

from step_04_hit_selection.accelerated_docking import AcceleratedDocking
from step_04_hit_selection.run_hit_selection import optimize_docking_performance
from utils.logger import LOGGER as logger

app = Flask(__name__)

# Глобальные переменные для состояния
current_job = None
job_results = {}

# HTML шаблон
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>GPU Молекулярный Докинг</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .status { background: #e8f5e8; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #4CAF50; }
        .gpu-info { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .error { background: #ffebee; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #f44336; }
        .success { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #4CAF50; }
        .warning { background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #ffc107; }
        button { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; margin: 10px 5px; }
        button:hover { background: #45a049; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        textarea { width: 100%; height: 150px; margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-family: monospace; }
        .form-group { margin: 20px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        .results { margin-top: 20px; }
        .result-item { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border: 1px solid #dee2e6; }
        .loading { text-align: center; color: #666; margin: 20px 0; }
        .metric { display: inline-block; margin: 0 20px; padding: 10px; background: #f1f1f1; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 GPU Молекулярный Докинг</h1>
        
        <div class="status">
            <h3>📊 Статус Системы</h3>
            <div class="metric">
                <strong>GPU:</strong> <span id="gpu-status">{{ gpu_status }}</span>
            </div>
            <div class="metric">
                <strong>Движок:</strong> <span id="gpu-engine">{{ gpu_engine }}</span>
            </div>
            <div class="metric">
                <strong>Активная задача:</strong> <span id="current-job">{{ current_job or 'Нет' }}</span>
            </div>
        </div>

        <div class="form-group">
            <h3>🧪 Запуск Докинга</h3>
            <label for="molecules">Молекулы (JSON формат):</label>
            <textarea id="molecules" placeholder='[{"id": "ethanol", "smiles": "CCO"}, {"id": "methanol", "smiles": "CO"}]'>{{ sample_molecules }}</textarea>
            <button onclick="startDocking()" id="dock-btn">Запустить Докинг</button>
            <button onclick="loadStatus()" id="status-btn">Обновить Статус</button>
        </div>

        <div id="results" class="results"></div>

        <div class="form-group">
            <h3>📋 Результаты Задач</h3>
            <div id="job-results"></div>
        </div>
    </div>

    <script>
        let currentJobId = null;
        let checkInterval = null;

        function showMessage(message, type = 'success') {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `<div class="${type}">${message}</div>`;
        }

        function showLoading(message) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `<div class="loading">⏳ ${message}</div>`;
        }

        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('gpu-status').textContent = data.gpu_available ? 'Доступен' : 'Недоступен';
                document.getElementById('gpu-engine').textContent = data.gpu_engine;
                document.getElementById('current-job').textContent = data.current_job || 'Нет';
                
                if (data.current_job) {
                    document.getElementById('dock-btn').disabled = true;
                    if (!checkInterval) {
                        checkInterval = setInterval(checkJobStatus, 2000);
                    }
                } else {
                    document.getElementById('dock-btn').disabled = false;
                    if (checkInterval) {
                        clearInterval(checkInterval);
                        checkInterval = null;
                    }
                }
            } catch (error) {
                showMessage('Ошибка загрузки статуса: ' + error.message, 'error');
            }
        }

        async function startDocking() {
            const molecules = document.getElementById('molecules').value;
            
            if (!molecules.trim()) {
                showMessage('Пожалуйста, введите молекулы', 'error');
                return;
            }

            try {
                const moleculesData = JSON.parse(molecules);
                if (!Array.isArray(moleculesData) || moleculesData.length === 0) {
                    showMessage('Молекулы должны быть массивом с элементами', 'error');
                    return;
                }

                showLoading('Запуск докинга...');
                
                const response = await fetch('/api/dock', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ molecules: moleculesData })
                });

                const data = await response.json();
                
                if (data.success) {
                    currentJobId = data.job_id;
                    showMessage(`Докинг запущен успешно! ID задачи: ${data.job_id}`, 'success');
                    setTimeout(loadStatus, 1000);
                } else {
                    showMessage('Ошибка: ' + data.error, 'error');
                }
            } catch (error) {
                showMessage('Ошибка: ' + error.message, 'error');
            }
        }

        async function checkJobStatus() {
            if (!currentJobId) return;

            try {
                const response = await fetch(`/api/job/${currentJobId}`);
                const data = await response.json();
                
                if (data.success && data.status) {
                    if (data.status.status === 'completed') {
                        showMessage(`Докинг завершен! Результатов: ${data.status.results_count || 0}`, 'success');
                        loadJobResults();
                        currentJobId = null;
                        loadStatus();
                    } else if (data.status.status === 'failed') {
                        showMessage(`Докинг завершился с ошибкой: ${data.status.error}`, 'error');
                        currentJobId = null;
                        loadStatus();
                    } else {
                        showLoading(`Докинг выполняется... (${data.status.status})`);
                    }
                }
            } catch (error) {
                console.error('Ошибка проверки статуса:', error);
            }
        }

        async function loadJobResults() {
            try {
                const response = await fetch('/api/results');
                const data = await response.json();
                
                const resultsDiv = document.getElementById('job-results');
                if (data.success && Object.keys(data.results).length > 0) {
                    resultsDiv.innerHTML = '<h4>Последние результаты:</h4>' + 
                        Object.entries(data.results).map(([jobId, results]) => `
                            <div class="result-item">
                                <strong>Задача:</strong> ${jobId}<br>
                                <strong>Результаты:</strong> ${Object.keys(results).length} молекул<br>
                                <details>
                                    <summary>Показать детали</summary>
                                    <pre>${JSON.stringify(results, null, 2)}</pre>
                                </details>
                            </div>
                        `).join('');
                } else {
                    resultsDiv.innerHTML = '<p>Нет результатов</p>';
                }
            } catch (error) {
                console.error('Ошибка загрузки результатов:', error);
            }
        }

        // Инициализация
        document.addEventListener('DOMContentLoaded', function() {
            loadStatus();
            loadJobResults();
        });

        // Обновление статуса каждые 10 секунд
        setInterval(loadStatus, 10000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Главная страница"""
    try:
        # Получаем статус системы
        config = optimize_docking_performance()
        gpu_available = False
        gpu_engine = "unknown"
        if config:
            gpu_available = config.get("use_gpu", False)
            gpu_engine = config.get("gpu_engine", "unknown")

        sample_molecules = json.dumps(
            [{"id": "ethanol", "smiles": "CCO"}, {"id": "methanol", "smiles": "CO"}, {"id": "water", "smiles": "O"}], indent=2
        )

        return render_template_string(
            HTML_TEMPLATE,
            gpu_status="Доступен" if gpu_available else "Недоступен",
            gpu_engine=gpu_engine,
            current_job=current_job,
            sample_molecules=sample_molecules,
        )
    except Exception as e:
        logger.error(f"Ошибка главной страницы: {e}")
        return f"Ошибка: {e}", 500


@app.route("/api/status")
def api_status():
    """API для получения статуса GPU системы"""
    try:
        config = optimize_docking_performance()

        # Получаем информацию о GPU
        gpu_info = []
        try:
            import GPUtil  # type: ignore

            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append(
                    {
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "memory_free": gpu.memoryFree,
                        "utilization": gpu.load * 100,
                    }
                )
        except ImportError:
            pass

        # Красивые названия для движков
        engine_names = {
            "vina_optimized": "CPU-Optimized Vina (32 Threads)",
            "autodock_gpu": "AutoDock-GPU (NVIDIA)",
        }

        gpu_engine = "unknown"
        if config and hasattr(config, "get"):
            gpu_engine = config.get("gpu_engine", "unknown")

        return jsonify(
            {
                "success": True,
                "gpu_available": len(gpu_info) > 0,
                "gpu_devices": gpu_info,
                "gpu_engine": engine_names.get(gpu_engine, gpu_engine),
                "job_count": len(job_results),
                "current_job": current_job,
            }
        )
    except Exception as e:
        logger.error(f"Ошибка API статуса: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/dock", methods=["POST"])
def api_dock():
    """API запуска докинга"""
    global current_job

    try:
        if current_job is not None:
            return jsonify({"success": False, "error": f"Другая задача уже выполняется: {current_job}"}), 400

        data = request.json
        molecules = data.get("molecules", [])

        if not molecules:
            return jsonify({"success": False, "error": "Нет молекул для докинга"}), 400

        # Генерируем ID задачи
        job_id = f"job_{int(time.time())}"
        current_job = job_id

        logger.info(f"Запуск докинга для {len(molecules)} молекул")

        # Создаем движок и запускаем докинг
        config = optimize_docking_performance()
        docking_engine = AcceleratedDocking(config)

        start_time = time.time()
        results = docking_engine.dock_molecules_batch(molecules)
        end_time = time.time()

        # Сохраняем результаты
        job_results[job_id] = {
            "results": results,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "molecules_count": len(molecules),
            "results_count": len(results),
        }

        current_job = None

        return jsonify(
            {
                "success": True,
                "job_id": job_id,
                "results_count": len(results),
                "duration": end_time - start_time,
                "message": f"Докинг завершен для {len(results)} молекул за {end_time - start_time:.2f} сек",
            }
        )

    except Exception as e:
        logger.error(f"Ошибка докинга: {e}")
        current_job = None
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/job/<job_id>")
def api_job_status(job_id):
    """API статуса задачи"""
    try:
        if job_id == current_job:
            return jsonify({"success": True, "status": {"status": "running", "message": "Задача выполняется..."}})

        if job_id in job_results:
            job_data = job_results[job_id]
            return jsonify(
                {
                    "success": True,
                    "status": {
                        "status": "completed",
                        "results_count": job_data["results_count"],
                        "duration": job_data["duration"],
                        "molecules_count": job_data["molecules_count"],
                    },
                }
            )

        return jsonify({"success": False, "error": "Задача не найдена"}), 404

    except Exception as e:
        logger.error(f"Ошибка статуса задачи: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/results")
def api_results():
    """API результатов всех задач"""
    try:
        # Возвращаем только результаты докинга
        results = {}
        for job_id, job_data in job_results.items():
            results[job_id] = job_data["results"]

        return jsonify({"success": True, "results": results})

    except Exception as e:
        logger.error(f"Ошибка получения результатов: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    logger.info("🚀 Запуск GPU Docking Flask Server...")
    logger.info("Сервер доступен на http://localhost:5000")

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
