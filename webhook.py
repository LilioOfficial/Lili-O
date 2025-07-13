from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import base64
import json
from pathlib import Path

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connection established")

    try:
        while True:
            message = await websocket.receive_text()
            try:
                ws_message = json.loads(message)

                if ws_message.get("event") == "audio_mixed_raw.data":
                    data = ws_message["data"]
                    buffer_b64 = data["data"]["buffer"]
                    recording_id = data["recording"]["id"]
                    file_path = Path(f"/tmp/{recording_id}.bin")

                    # Decode base64 audio
                    decoded_bytes = base64.b64decode(buffer_b64)

                    # Append to file
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    with file_path.open("ab") as f:
                        f.write(decoded_bytes)

                    print(f"🔊 Wrote audio chunk to {file_path}")
                else:
                    print(f"⚠️ Unhandled event type: {ws_message.get('event')}")

            except Exception as e:
                print(f"❌ Error processing message: {e}")

    except WebSocketDisconnect:
        print("❎ WebSocket disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
