"""
WebSocket endpoint for real-time updates
"""

import json
import asyncio
from typing import Dict, Set, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.websockets import WebSocketState
from jose import jwt, JWTError

try:
    from backend.api.auth import SECRET_KEY, ALGORITHM
except ImportError:
    SECRET_KEY = "test-secret-key"
    ALGORITHM = "HS256"

router = APIRouter()

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.job_subscriptions: Dict[str, Set[str]] = {}  # job_id -> set of user_ids
        
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        
    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                
    async def send_to_user(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[user_id]:
                try:
                    if connection.application_state == WebSocketState.CONNECTED:
                        await connection.send_json(message)
                    else:
                        disconnected.add(connection)
                except:
                    disconnected.add(connection)
            
            # Clean up disconnected sockets
            for conn in disconnected:
                self.active_connections[user_id].discard(conn)
                
    async def send_job_update(self, job_id: str, update: dict):
        """Send job update to all subscribed users"""
        if job_id in self.job_subscriptions:
            for user_id in self.job_subscriptions[job_id]:
                await self.send_to_user(user_id, {
                    "type": "job_update",
                    "job_id": job_id,
                    **update
                })
                
    def subscribe_to_job(self, user_id: str, job_id: str):
        if job_id not in self.job_subscriptions:
            self.job_subscriptions[job_id] = set()
        self.job_subscriptions[job_id].add(user_id)
        
    def unsubscribe_from_job(self, user_id: str, job_id: str):
        if job_id in self.job_subscriptions:
            self.job_subscriptions[job_id].discard(user_id)
            if not self.job_subscriptions[job_id]:
                del self.job_subscriptions[job_id]

# Global connection manager
manager = ConnectionManager()

# Helper function to verify token
async def verify_websocket_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None

@router.websocket("/progress")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...)
):
    """WebSocket endpoint for real-time progress updates"""
    
    # Verify token
    username = await verify_websocket_token(token)
    if not username:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    # For now, use username as user_id
    user_id = username
    
    # Connect
    await manager.connect(websocket, user_id)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "Connected to NeuralForge Studio"
        })
        
        # Listen for messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message["type"] == "subscribe":
                job_id = message.get("job_id")
                if job_id:
                    manager.subscribe_to_job(user_id, job_id)
                    await websocket.send_json({
                        "type": "subscribed",
                        "job_id": job_id
                    })
                    
            elif message["type"] == "unsubscribe":
                job_id = message.get("job_id")
                if job_id:
                    manager.unsubscribe_from_job(user_id, job_id)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "job_id": job_id
                    })
                    
            elif message["type"] == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": message.get("timestamp")
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, user_id)

# Export manager for use in other modules
def get_connection_manager():
    return manager