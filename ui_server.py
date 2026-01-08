from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import threading
from state import ui_state, ui_lock
import copy

app = FastAPI()

# === Keep track of all connected clients ===
connections = []

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("[UI] Client connected")

    # Add client to global list
    connections.append(ws)
    try:
        while True:
            # Just wait for pings or messages from client (optional)
            msg = await ws.receive_text()
            # Optionally handle client messages here
    except WebSocketDisconnect:
        print("[UI] Client disconnected")
        connections.remove(ws)
