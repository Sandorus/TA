from fastapi import FastAPI, WebSocket
import asyncio
import threading
import time

from state import ui_state, ui_lock
import copy


app = FastAPI()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("[UI] Client connected")

    while True:
        with ui_lock:
            snapshot = copy.deepcopy(ui_state)

        #print("[UI] Sending snapshot", snapshot)
        await ws.send_json(snapshot)
        await asyncio.sleep(0.2)