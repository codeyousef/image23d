"""Real-time progress display component using WebSocket"""

import gradio as gr
from typing import Any, Optional, Tuple, List


def create_progress_component() -> gr.HTML:
    """Create a real-time progress display component
    
    Returns:
        Progress display HTML component
    """
    
    # HTML template with embedded JavaScript for WebSocket connection
    progress_html = """
    <div id="progress-container" style="position: fixed; bottom: 20px; right: 20px; width: 400px; max-height: 500px; z-index: 1000;">
        <div id="progress-widget" style="background: #fff; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); overflow: hidden;">
            <!-- Header -->
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 15px; background: #2196f3; color: white; cursor: pointer;" onclick="toggleProgress()">
                <span style="font-weight: bold;">⚡ Generation Progress</span>
                <div>
                    <span id="active-tasks">0</span> active
                    <button id="minimize-btn" style="background: none; border: none; color: white; margin-left: 10px; cursor: pointer;">▼</button>
                </div>
            </div>
            
            <!-- Content -->
            <div id="progress-content" style="max-height: 400px; overflow-y: auto;">
                <div id="progress-logs" style="padding: 10px;">
                    <p style="color: #666; text-align: center;">Waiting for tasks...</p>
                </div>
            </div>
        </div>
    </div>
    
    <style>
    .progress-entry {
        margin-bottom: 10px;
        padding: 8px;
        border-left: 3px solid #ddd;
        background: #f5f5f5;
        border-radius: 0 4px 4px 0;
    }
    
    .progress-entry.info { border-left-color: #2196f3; }
    .progress-entry.success { border-left-color: #4caf50; background: #e8f5e9; }
    .progress-entry.warning { border-left-color: #ff9800; background: #fff3e0; }
    .progress-entry.error { border-left-color: #f44336; background: #ffebee; }
    
    .progress-bar {
        width: 100%;
        height: 20px;
        background: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 5px 0;
    }
    
    .progress-bar-fill {
        height: 100%;
        background: #2196f3;
        transition: width 0.3s ease;
    }
    
    .task-header {
        font-weight: bold;
        margin-bottom: 5px;
        display: flex;
        justify-content: space-between;
    }
    
    .timestamp {
        font-size: 0.8em;
        color: #666;
    }
    
    #progress-widget.minimized #progress-content {
        display: none;
    }
    
    #progress-widget.minimized {
        width: 200px;
    }
    </style>
    
    <script>
    let ws = null;
    let reconnectInterval = null;
    let activeTasks = {};
    let isMinimized = false;
    
    let currentPort = 8765;
    const ports = [8765, 8766, 8767, 8768, 8769];
    let portIndex = 0;
    
    function connectWebSocket() {
        if (ws && ws.readyState === WebSocket.OPEN) return;
        
        ws = new WebSocket(`ws://localhost:${currentPort}`);
        
        ws.onopen = function() {
            console.log(`WebSocket connected on port ${currentPort}`);
            clearInterval(reconnectInterval);
            portIndex = 0;  // Reset port index on success
            updateStatus('Connected', 'success');
        };
        
        ws.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                handleMessage(data);
            } catch (e) {
                console.error('Failed to parse message:', e);
            }
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            updateStatus('Connection error', 'error');
        };
        
        ws.onclose = function() {
            console.log(`WebSocket disconnected from port ${currentPort}`);
            
            // Try next port if connection failed immediately
            if (ws.readyState === WebSocket.CLOSED && portIndex < ports.length - 1) {
                portIndex++;
                currentPort = ports[portIndex];
                console.log(`Trying port ${currentPort}...`);
                setTimeout(connectWebSocket, 1000);
            } else {
                // All ports tried or normal disconnect
                updateStatus('Disconnected', 'warning');
                // Reconnect after 5 seconds, starting from first port
                portIndex = 0;
                currentPort = ports[0];
                reconnectInterval = setInterval(connectWebSocket, 5000);
            }
        };
    }
    
    function handleMessage(data) {
        if (data.type === 'batch') {
            // Handle batch of messages
            data.messages.forEach(msg => processUpdate(msg));
        } else if (data.type === 'task_state') {
            // Handle task state update
            updateTaskState(data.data);
        } else if (data.type === 'heartbeat') {
            // Keep connection alive
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'pong'}));
            }
        }
    }
    
    function processUpdate(update) {
        const taskId = update.task_id;
        
        // Initialize task if new
        if (!activeTasks[taskId]) {
            activeTasks[taskId] = {
                id: taskId,
                type: update.data.task_type || 'unknown',
                logs: [],
                progress: 0,
                status: 'running'
            };
        }
        
        const task = activeTasks[taskId];
        
        // Handle different update types
        switch(update.type) {
            case 'progress':
                task.progress = update.data.progress || 0;
                task.currentStep = update.data.message || '';
                addLogEntry(taskId, update.data.message, update.level, task.progress);
                break;
                
            case 'log':
            case 'info':
            case 'warning':
            case 'error':
                addLogEntry(taskId, update.data.message, update.level);
                break;
                
            case 'success':
                task.status = 'completed';
                task.progress = 1.0;
                addLogEntry(taskId, 'Task completed successfully!', 'success', 1.0);
                setTimeout(() => removeTask(taskId), 30000); // Remove after 30s
                break;
                
            case 'error':
                task.status = 'failed';
                addLogEntry(taskId, update.data.error || 'Task failed', 'error');
                break;
                
            case 'preview':
                // Handle preview data
                showPreview(taskId, update.data);
                break;
        }
        
        updateDisplay();
    }
    
    function addLogEntry(taskId, message, level, progress) {
        const task = activeTasks[taskId];
        if (!task) return;
        
        const entry = {
            message: message,
            level: level || 'info',
            timestamp: new Date().toLocaleTimeString(),
            progress: progress
        };
        
        task.logs.push(entry);
        
        // Keep only last 50 logs per task
        if (task.logs.length > 50) {
            task.logs.shift();
        }
    }
    
    function updateDisplay() {
        const container = document.getElementById('progress-logs');
        const activeCount = Object.keys(activeTasks).length;
        
        document.getElementById('active-tasks').textContent = activeCount;
        
        if (activeCount === 0) {
            container.innerHTML = '<p style="color: #666; text-align: center;">No active tasks</p>';
            return;
        }
        
        let html = '';
        
        for (const taskId in activeTasks) {
            const task = activeTasks[taskId];
            const shortId = taskId.substring(0, 8);
            
            html += `
                <div class="task-section" style="margin-bottom: 15px; padding: 10px; background: white; border-radius: 4px;">
                    <div class="task-header">
                        <span>${task.type} - ${shortId}</span>
                        <span class="timestamp">${task.status}</span>
                    </div>
            `;
            
            // Add progress bar if applicable
            if (task.progress > 0) {
                html += `
                    <div class="progress-bar">
                        <div class="progress-bar-fill" style="width: ${task.progress * 100}%"></div>
                    </div>
                `;
            }
            
            // Add recent logs
            const recentLogs = task.logs.slice(-5);
            for (const log of recentLogs) {
                html += `
                    <div class="progress-entry ${log.level}">
                        <div style="display: flex; justify-content: space-between;">
                            <span>${log.message}</span>
                            <span class="timestamp">${log.timestamp}</span>
                        </div>
                    </div>
                `;
            }
            
            html += '</div>';
        }
        
        container.innerHTML = html;
        
        // Auto-scroll to bottom
        const content = document.getElementById('progress-content');
        content.scrollTop = content.scrollHeight;
    }
    
    function removeTask(taskId) {
        delete activeTasks[taskId];
        updateDisplay();
    }
    
    function toggleProgress() {
        isMinimized = !isMinimized;
        const widget = document.getElementById('progress-widget');
        const btn = document.getElementById('minimize-btn');
        
        if (isMinimized) {
            widget.classList.add('minimized');
            btn.textContent = '▲';
        } else {
            widget.classList.remove('minimized');
            btn.textContent = '▼';
        }
    }
    
    function updateStatus(message, level) {
        const container = document.getElementById('progress-logs');
        if (Object.keys(activeTasks).length === 0) {
            container.innerHTML = `<p style="color: ${level === 'error' ? 'red' : '#666'}; text-align: center;">${message}</p>`;
        }
    }
    
    function showPreview(taskId, previewData) {
        // Handle preview display (e.g., show thumbnail)
        console.log('Preview for task', taskId, previewData);
    }
    
    // Initialize WebSocket connection
    connectWebSocket();
    
    // Clean up on page unload
    window.addEventListener('beforeunload', function() {
        if (ws) {
            ws.close();
        }
    });
    </script>
    """
    
    return gr.HTML(progress_html, visible=True)


def create_simple_progress_bar() -> Tuple[gr.Progress, gr.HTML]:
    """Create a simple progress bar for fallback
    
    Returns:
        Tuple of (progress bar, info text)
    """
    progress = gr.Progress()
    info = gr.HTML()
    
    return progress, info