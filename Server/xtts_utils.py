import requests
import json
import re

from const import XTTS_URL, SPEAKER_WAV_PATH


def split_and_merge_sentence(text, threshold=30):
    """
    Split sentences based on punctuation marks and merge short sentences to ensure each segment is as close to or exceeds the threshold length.
    :param text: Input text
    :param threshold: Length threshold
    :return: List of merged sentences
    """
    # Define separators (Chinese and English commas, periods, etc.)
    separators = r'([,，.。!！?？;；])'
    
    # Split the text using regex, preserving separators
    segments = re.split(separators, text)
    
    # Combine separators with preceding text
    combined_segments = []
    temp = ""
    
    # Merge separators with the preceding text
    for i in range(0, len(segments)):
        temp += segments[i]
        # If it's a separator
        if i % 2 == 1:
            combined_segments.append(temp)
            temp = ""
    
    # Handle any remaining text
    if temp:
        combined_segments.append(temp)
    
    # Merge short sentences to ensure each segment is as close to or exceeds the threshold
    result = []
    current_segment = ""
    
    for segment in combined_segments:
        if not current_segment:
            current_segment = segment
        elif len(current_segment) + len(segment) <= threshold:
            # If the merged segment is still below the threshold, continue merging
            current_segment += segment
        else:
            # If the current segment exceeds the threshold, save it and start a new segment
            if current_segment:
                result.append(current_segment)
            current_segment = segment
    
    # Handle the last segment
    if current_segment:
        # If the last segment is too short, try merging it with the previous segment
        if len(current_segment) < threshold and result:
            last = result.pop()
            result.append(last + current_segment)
        else:
            result.append(current_segment)
    
    return result

def generate_tts(text):
    # Set the API URL
    tts_url = XTTS_URL
    lang_tts = 'en'
    speaker_wav = SPEAKER_WAV_PATH
    data = {
        "text": text,  # Text to convert
        "speaker_wav": speaker_wav,  # Path to the audio file
        "language": lang_tts,  # Language code
    }
    # Convert data to JSON and send a POST request
    response = requests.post(tts_url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

    return response.content

def generate_tts_opt_stream(sentence):
    # Remove unnecessary matching and extract sentence content
    text = sentence.strip().replace('"', '')
    #print(f"Processing text: {text}")
    # Split text into smaller parts (for better audio chunking)
    parts = split_and_merge_sentence(text, 25)
    parts = [p.strip() for p in parts if p.strip()]  # Clean empty parts
    #print(f"Processed text: {parts}")
    
    for part in parts:
        print('Part:', part)
        yield generate_tts(part)