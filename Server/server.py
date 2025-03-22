######################### Update Log ###################################
# Version: 2025/2/1
# Modified the model invocation method to use Ollama, activating and testing Ollama when the server program starts.

# Version: 2024/12/24
# Added functionality to recognize only specified license plate numbers and vehicle colors.

# Version: 2024/12/22
# Added depth estimation and license plate recognition features, completing the full system construction.

# Version: 2024/12/19
# Initial setup of the server-side workflow.

####################### End of Update Log ##############################


import asyncio
import json
import websockets
import os
import base64
import time
import ollama
import subprocess
from time import time as ttime
from uuid import uuid4
from PIL import Image
import cv2
import numpy as np
import shutil


from const import *
from clientsession import ClientSession
from xtts_utils import generate_tts_opt_stream
from depthEstimation import DepthEstimation

"""
conda activate VisionRide

cd ~

netstat -lnp|grep 11033

python server.py

nohup python -u server.py >> /home/ubuntu/server.log 2>&1 &
"""

# Load the model
print("model loading...")

server_processing = False  # Global lock to indicate whether the server is processing user input, preventing multi-threading resource conflicts.

LLM_model = 'phi4'  # Model ID deployed on Ollama.
# Activate Ollama
print('model test question:', "Hello, who are you?")
# Chat mode
response = ollama.chat(model=LLM_model, messages=[
    {"role": "user", "content": "Hello, who are you?"}
])
print("model test response:", response["message"]["content"])

# Create a folder to store temporary files
os.makedirs('./tmp_files', exist_ok=True)

# Load the depth estimation model
depth_model = DepthEstimation('indoor', 'cuda')

print("model loaded!")
# Model loading completed.


def checkPlateNumber(obj_info):
    """
    Check if a specified license plate number is detected.

    Params:
    obj_info (dict): License plate data.

    Returns:
    The index of the specified license plate in obj_info['results'], or -1 if not found (int).
    """
    if len(obj_info['results']) == 0:
        return -1
    if len(PLATE_NUMBERS) == 0:
        return 0
    for ind, result in enumerate(obj_info['results']):
        for candidate in result['candidates']:
            if candidate['plate'] in PLATE_NUMBERS:
                return ind
    return -1


def genDepthGraph(photo_path):
    """
    Perform image depth estimation.

    Params:
    photo_path (str): Path to the image file.

    Returns:
    A 2D depth map matching the image size, with depth values in meters (np.ndarray).
    """
    image = Image.open(photo_path).convert("RGB")
    depth_graph = depth_model.generate(image)
    return depth_graph


def genObjInfo(photo_path, oper_dir_path):
    """
    Perform license plate recognition.

    Params:
    photo_path (str): Path to the image file.
    oper_dir_path (str): Path to store temporary files.

    Returns:
    Recognized license plate data (dict).
    """
    output_path = os.path.join(oper_dir_path, 'objInfo.json')
    with open(output_path, 'w') as f:
        subprocess.run(['alpr', '-j', '-p', 'nj', photo_path], stdout=f)
    with open(output_path) as f:
        objInfo = json.load(f)
    for res in objInfo['results']:
        print(f"Detected plate number: {res['plate']}")
    return objInfo


def genObjDepth(plate_result, depth_graph):
    """
    Calculate the depth of the license plate.

    Params:
    plate_result (dict): Data of the specified license plate.
    depth_graph (np.array): Depth map.

    Returns:
    The average depth of the license plate (float).
    """
    # Get the coordinates for four vertices
    coordinates = plate_result["coordinates"]
    top_left = coordinates[0]
    top_right = coordinates[1]
    bottom_right = coordinates[2]
    bottom_left = coordinates[3]

    Vertexes = np.array([[top_left["x"], top_left["y"]], [top_right["x"], top_right["y"]], 
                        [bottom_right["x"], bottom_right["y"]], [bottom_left["x"], bottom_left["y"]]], dtype=np.int32)

    # A new blank matrix that has the same size as the depth matrix
    mask = np.zeros_like(depth_graph, dtype=np.uint8)

    cv2.fillPoly(mask, [Vertexes], 255)

    depth_region = np.ma.masked_where(mask == 0, depth_graph)

    average_depth = np.mean(depth_region)
    return average_depth



def genPrompt(plate_result, image_shape, obj_depth):
    """
    Generate a prompt based on license plate information, image size, and license plate depth.

    Params:
    plate_result (dict): Data of the specified license plate.
    image_shape (tuple): A tuple (x, y) representing the image dimensions.
    obj_depth (float): Depth of the license plate.

    Returns:
    A prompt for LLM input (str).
    """
    if len(PLATE_NUMBERS) == 0:
        plate = plate_result['plate']  # Use the detected license plate number if none is specified.
    else:
        plate = PLATE_NUMBERS[0]  # Use the specified license plate number, assuming the first in the list is correct.
    car_color = CAR_COLOR
    coords = plate_result['coordinates']
    total_x = sum(point['x'] for point in coords)
    num_points = len(coords)
    midpoint_x = total_x / num_points
    max_x = image_shape[0]
    if midpoint_x < max_x/3:
        loc = 'left'
    elif midpoint_x < max_x*2/3:
        loc = 'middle'
    else:
        loc = 'right'
    if car_color is None:
        prompt = f'The vehicle with license plate number {plate} is located {str(int(obj_depth))} meters in front on the {loc}.'
    else:
        prompt = f'The vehicle with license plate number {plate} and {car_color} color is located {str(int(obj_depth))} meters in front on the {loc}.'
    return prompt


# Read the Base64 encoded content and decode it to save as a file
def decode_base64_to_file(data, output_file_path):
    file_content = base64.b64decode(data)
    with open(output_file_path, 'wb') as output_file:
        output_file.write(file_content)


async def client_handler(websocket):
    global server_processing
    session = ClientSession(websocket)
    print('---> client_handler：', id(session), flush=True)
    
    websocket = session.websocket
    assert id(websocket)==id(session.websocket), "id(websocket) != id(session.websocket)"
    
    try:
        async for message in session.websocket:
            
            # Process the configuration information
            if isinstance(message, str):
                message_json = json.loads(message)
                msgType = message_json['type']
                
                if msgType == 'init':
                    ## `connect`: indicates the initial configuration information sent by the client
                    print('-----> Received meg: [init]', flush=True)
                    ## Reset historical messages and other records
                    session.reset()
                    session.websocket = load_config(session, message_json)
                    server_processing = True
                    await gen_and_send_tts(session, WELCOME_WORDS)
                    server_processing = False
                elif msgType == 'status':
                    ## `audio_playing`: indicates that the client is starting/ending the playback of audio
                    print('-----> Received meg: [status]: ', message_json["audio_playing"], flush=True)
                    session.client_is_playing_audio = int(message_json["audio_playing"])
                elif msgType == 'image':
                    # Image process
                    if not session.client_is_playing_audio and not server_processing:
                        server_processing = True
                        tic_total = ttime()
                        oper_id = str(uuid4())
                        oper_dir_path = os.path.join('./tmp_files', oper_id)
                        os.makedirs(oper_dir_path, exist_ok=True)
                        photo_path = os.path.join(oper_dir_path, 'photo.jpg')
                        decode_base64_to_file(message_json['data'], photo_path)
                        obj_info = genObjInfo(photo_path, oper_dir_path)
                        result_index = checkPlateNumber(obj_info)
                        if result_index < 0:    # The specified license plate number was not detected
                            server_processing = False
                            shutil.rmtree(oper_dir_path)
                            print('No plate detected.')
                            continue
                        depth_image = genDepthGraph(photo_path)
                        obj_depth = genObjDepth(obj_info['results'][result_index], depth_image)
                        prompt = genPrompt(obj_info['results'][result_index], depth_image.shape, obj_depth)
                        await llm_and_tts(session, prompt)
                        print('----> 【耗时测试】总生成时间: ', round(ttime()-tic_total, 2), flush=True)
                        server_processing = False
                        shutil.rmtree(oper_dir_path)

    except websockets.exceptions.ConnectionClosedError:
        print("ConnectionClosed... Session:", id(session), flush=True)
        await ws_reset(session)
        # del session
    except session.websocket.InvalidState:
        print("InvalidState...")
    except Exception as e:
        print("Exception:", e)


async def ws_reset(session):
    await session.websocket.close()


def load_config(session, messagejson):
    print('---> `load_config` websocket id is: ', id(session.websocket), flush=True)
    websocket = session.websocket
    assert id(websocket) == id(session.websocket), "session.websocket is not the same as websocket"
    
    return websocket


async def gen_and_send_tts(session, text):
    tic_tts = time.time()
    await send_text_to_client(
            session.websocket,
            json.dumps({
                'type': 'audioBegin'
            })
        )
    tts_stream = generate_tts_opt_stream(text)
    for i, audio_content in enumerate(tts_stream):

        await send_audio_to_client_bytes(session.websocket, audio_content)
    await send_text_to_client(
            session.websocket,
            json.dumps({
                'type': 'audioEnd'
            })
        )
    tictoc_tts = time.time()
    print('----> 【Time Consumption Test】TTS Streaming Generation & Transmission:', round(tictoc_tts-tic_tts, 2), 's')


async def llm_and_tts(session, prompt):
    print('----> llm_and_tts [prompt]', prompt, flush=True)
    ##############  LLM operation  ##############
    tic = time.time()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]    # Add system prompt
    messages.append({"role": "user", "content": prompt})    # Add user input
    response = ollama.chat(
        model=LLM_model,  # Model name
        messages=messages
    )
    response = response["message"]["content"]

    toc = time.time()
    # Send the LLM's text response to the client
    await send_text_to_client(
        session.websocket,
        json.dumps({
            'type': 'text',
            'src': 'llm_response',
            'text': response
        })
    )
    tictoc = time.time()
    print('---> LLM response sent:', f"【{response}】")
    print('----> 【Time test】 LLM generation:', round(toc-tic, 2), 's', ' | Transmission:', round(tictoc-toc, 2), 's')
    
    ## If the response is empty, return immediately
    if response == '':  
        await send_text_to_client(
            session.websocket,
            json.dumps({
                'type': 'noResponse'
            })
        )
        return  

    
    ##############  TTS operation  ##############
    await gen_and_send_tts(session, response)


async def send_text_to_client(text_client_websocket, message):
    try:
        print('---> server `send_text_to_client`: text_client_websocket is None', 
            text_client_websocket is None)

        if text_client_websocket:
            await text_client_websocket.send(message)
        else:
            print("!!!!!! > text_client_websocket is None")
    except Exception as e:
        print(f"Failed to send text to the client: send_text_to_client: {str(e)}")


async def send_audio_to_client_bytes(
        audio_client_websocket, 
        audio_data, 
        audio_format='wav'
    ):

    import wave
    output_filename = "output.wav"

    print(len(audio_data))
    # Configure WAV file parameters
    sample_rate = 16000  # Sampling rate
    num_channels = 1     # Mono channel
    sample_width = 2     # Bytes per sample (typically 2 for 16-bit PCM)

    # Create and write to the WAV file
    with wave.open(output_filename, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)         # Set the number of channels
        wav_file.setsampwidth(sample_width)         # Set the bytes per sample
        wav_file.setframerate(sample_rate)          # Set the sampling rate
        wav_file.writeframes(audio_data)            # Write the audio data

    audio_data_base64 = base64.b64encode(audio_data).decode('utf-8')
    message = json.dumps({
        'type': 'audio', 
        'format': audio_format,
        'audioData': audio_data_base64
    })

    # Send message through WebSocket
    try:
        if audio_client_websocket:
            await audio_client_websocket.send(message)
        else:
            print("!!!!!! > audio_client_websocket is None")
    except Exception as e: 
        print(f"Failed to send audio to the client: send_audio_to_client_bytes: {str(e)}")


async def start_server():
    while True:
        try:
            # Create WebSocket server
            async with websockets.serve(client_handler, '0.0.0.0', 8002, max_size=100_000_000):
                print("✅ Server started, listening on 0.0.0.0:8002")
                await asyncio.Future()  # Run indefinitely
        except Exception as e:
            print(f"⚠️ Server crashed: {e}. Restarting in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(start_server())
