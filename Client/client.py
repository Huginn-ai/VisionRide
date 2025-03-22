"""
Client-side application for real-time photo capture and audio streaming via WebSocket.

Dependencies:
    pip install opencv-python
    pip install websockets
    pip install keyboard
"""

# Standard library imports
from config import *  # Contains HOST, PORT, and other configuration parameters
from photo_controller import PhotoController  # Camera controller module
from queue import Queue
import websockets
import asyncio
import base64
import json
import traceback
import sys
from asyncio.subprocess import PIPE

# Global message queue for outgoing WebSocket messages
messageQueue = Queue()
# Global audio queue for received audio data
audioQueue = Queue()

# Initialize camera controller
camera = PhotoController()

# Exit if camera is not available
if camera.is_available == False:
    sys.exit(1)

def encode_data_to_base64(data: bytes) -> str:
    """Encode binary data to Base64 string.
    
    Args:
        data: Binary data to encode
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(data).decode('utf-8')

async def take_photo():
    """Capture photos periodically and send them via WebSocket.
    
    Sends initialization message first, then captures frames at 1-second intervals.
    Handles keyboard interrupts for clean exit.
    """
    # Send initialization message
    init_message = json.dumps({"type": 'init'})
    messageQueue.put(init_message)

    try:
        while True:
            frame = camera.get_photo()
            if frame:
                # Encode and queue the image data
                data = encode_data_to_base64(frame)
                messageQueue.put(json.dumps({
                    "type": 'image',
                    "data": data
                }))
                print('Captured photo.')
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print('Program interrupted by user.')
        sys.exit()

async def myPing():
    """Maintain WebSocket connection by sending periodic pings."""
    global websocket
    while True:
        await websocket.ping()
        await asyncio.sleep(10)

async def message():
    """Handle incoming WebSocket messages.
    
    Processes different message types:
    - LLM responses: Prints text responses
    - Audio data: Decodes and queues for playback
    - Text messages: Currently just logs reception
    """
    try:
        while True:
            msg = await websocket.recv()
            msg_data = json.loads(msg)

            # Handle LLM responses
            if 'src' in msg_data and msg_data['src'] == 'llm_response':
                print('LLM Response:', msg_data["text"])
                continue

            # Handle audio data
            if msg_data.get("type", '') == 'audio':
                try:
                    audio_data = base64.b64decode(msg_data["audioData"])
                    audioQueue.put(audio_data)
                except Exception as e:
                    print(f'Audio decoding error: {e}')
                    traceback.print_exc()
                continue

            # Handle other text messages
            if msg_data.get("type", '') == 'text':
                print('Received text message:', msg_data)
                continue

            print('Unhandled message type:', msg_data.get("type", 'unknown'))

    except Exception as e:
        print("Message handling error:", e)
        traceback.print_exc()

async def send_message():
    """Send queued messages to WebSocket server.
    
    Continuously checks message queue and sends messages as they become available.
    """
    try:
        while True:
            while messageQueue.empty():
                await asyncio.sleep(0.1)
            data = messageQueue.get()
            await websocket.send(data)
    except Exception as e:
        print(f"Message sending error: {str(e)}")

async def play_audio():
    """Play received audio data using system's aplay command.
    
    Manages audio playing status and sends status updates via WebSocket.
    """
    await asyncio.sleep(2)  # Initial delay
    is_playing = False
    
    try:
        while True:
            # Update status when audio stops
            while audioQueue.empty():
                if is_playing:
                    is_playing = False
                    messageQueue.put(json.dumps(
                        {"type": 'status', "audio_playing": '0'}
                    ))
                    await asyncio.sleep(1)
                await asyncio.sleep(0.1)

            # Process audio data
            audio_data = audioQueue.get()
            if not is_playing:
                is_playing = True
                messageQueue.put(json.dumps(
                    {"type": 'status', "audio_playing": '1'}
                ))

            # Write to file and play using aplay
            with open('audio.wav', 'wb') as f:
                f.write(audio_data)
                
            process = await asyncio.create_subprocess_exec(
                "aplay", "audio.wav", stdout=PIPE, stderr=PIPE
            )
            await process.wait()

    except Exception as e:
        print(f"Audio playback error: {str(e)}")
        traceback.print_exc()

async def ws_client():
    """Main WebSocket client routine.
    
    Establishes connection and manages all asynchronous tasks:
    - Photo capture
    - Connection pinging
    - Message handling
    - Audio playback
    - Message sending
    """
    global websocket

    uri = f"ws://{HOST}:{PORT}"
    print(f"Connecting to server at {uri}")

    try:
        async with websockets.connect(
            uri, 
            subprotocols=["binary"], 
            ping_interval=None, 
            ssl=None, 
            max_size=100_000_000
        ) as websocket:
            # Create task group
            tasks = [
                asyncio.create_task(take_photo()),
                asyncio.create_task(myPing()),
                asyncio.create_task(message()),
                asyncio.create_task(play_audio()),
                asyncio.create_task(send_message()),
            ]

            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                print("\nTask execution error:")
                traceback.print_exc()
                # Cleanup tasks
                for task in tasks:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                raise

    except Exception as e:
        print("\nConnection failed:")
        traceback.print_exc()
        sys.exit(1)

def one_thread():
    """Run the WebSocket client in the main event loop."""
    try:
        asyncio.get_event_loop().run_until_complete(ws_client())
    except Exception as e:
        print("\nFatal error occurred:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    """Entry point for standalone execution."""
    one_thread()