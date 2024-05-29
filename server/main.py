# Flask
from flask import Flask, request, jsonify
from flask_cors import CORS

# LLM
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents import AgentType, initialize_agent
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from openai import OpenAI

# Others
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
#from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
#from classifiers import DeepFakeClassifier
from pytube import YouTube

import pytesseract
import os
import datetime
import requests
import re
import json
import uuid

import pandas as pd
from translate import Translator

client = OpenAI(api_key="OPENAI")

os.environ['GOOGLE_CSE_ID'] = os.environ.get('GOOGLE_CSE_ID', 'CSE')
os.environ['GOOGLE_API_KEY'] = os.environ.get('GOOGLE_API_KEY', 'API')

search = GoogleSearchAPIWrapper()

# Helper function - It was faster/easier to do everything is one folder, please forgive me...
def chunk_text(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def translate_to_georgian(text):
    chunk_size = 500
    chunks = chunk_text(text, chunk_size)
    translator = Translator(to_lang='ka')

    translated_chunks = [translator.translate(chunk) for chunk in chunks]
    return ''.join(translated_chunks)

def translate_to_english(text):
    chunk_size = 500
    chunks = chunk_text(text, chunk_size)
    translator = Translator(to_lang='en')
    translated_chunks = [translator.translate(chunk) for chunk in chunks]
    return ''.join(translated_chunks)

def download_video(video_url, target_dir):
    if "youtube.com" in video_url or "youtu.be" in video_url:
        # Handle YouTube URL
        yt = YouTube(video_url)
        video_stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
        if video_stream:
            video_path = os.path.join(target_dir, yt.title + '.mp4')
            video_stream.download(output_path=target_dir, filename=yt.title + '.mp4')
            return video_path
        else:
            raise Exception("No suitable video stream found")
    else:
        # Handle other URLs
        response = requests.get(video_url)
        if response.status_code == 200:
            video_path = os.path.join(target_dir, 'downloaded_video.mp4')
            with open(video_path, 'wb') as f:
                f.write(response.content)
            return video_path
        else:
            raise Exception("Failed to download video")

# Prediction function
def predict_video(video_url, test_dir):
    video_path = download_video(video_url, test_dir)

    frames_per_video = 32
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)
    input_size = 380
    strategy = confident_strategy

    test_videos = [os.path.basename(video_path)]

    predictions = predict_on_video_set(face_extractor=face_extractor, input_size=input_size, models=models,
                                       strategy=strategy, frames_per_video=frames_per_video, videos=test_videos,
                                       num_workers=6, test_dir=test_dir)

    classifications = ["DF" if label > 0.7 else "NDF" for label in predictions]

    return {"filename": test_videos, "label": predictions, "classification": classifications}

# Weights import from DVR (Deep video recognition)

#weights_dir = "weights"
#model_names = [
#    "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
#    "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
    # "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29",
    # "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31",
    # "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37",
    # "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40",
    # "final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23",
#]

#models = []
#for model_name in model_names:
#    path = os.path.join(weights_dir, model_name)
#    model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
#    print("loading state dict {}".format(path))
#    checkpoint = torch.load(path, map_location="cpu")
#    state_dict = checkpoint.get("state_dict", checkpoint)
#    model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
#    model.eval()
#    del checkpoint
#    models.append(model.half())

app = Flask(__name__)
CORS(app)

@app.route('/api/fact', methods=['POST'])
def receive_fact():
    data = request.get_json()
    if 'data' in data and 'url' in data and 'contentType' in data:
        url = data['url']
        content_type = data['contentType']

        if content_type == 'text':
            text = data['data']

            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": f"NOTE: You need to formulate questions like `Ronaldos current football team`. Current date: {datetime.datetime.now()}. You are smart AI, your goal is to generate google search query based on chunk of text to determine this text is true or not. NOTE: formulate question to determine it's true or no. never directly use date, and text. Always formulate questions. return ouput as json"},
                    {"role": "user", "content": "Ronaldo changed club to - Barcelona"},
                    {"role": "assistant", "content": "{'query': 'Ronaldos current football team'}"},
                    {"role": "user", "content": text}
                ]
            )

            output = json.loads(response.choices[0].message.content)
            if output['query']:
                searched_data = search.run(output['query'])

                evaluation_string = f"""
                                    Text to check: {text}
                                    Google output: {searched_data}

                                    Note: return strings in original language (Georgian or English) 
                                    Note: Keys must be in english only, just values might change
                                    """


                evaluation_request = client.chat.completions.create(
                    model="gpt-4o",
                    response_format={"type": "json_object"},
                    messages=[
                            {"role": "system", "content": f"Current date: {datetime.datetime.now()}. You are FACT checker AI. your goal is to check if provided text from image is true or no based on various factors. You need to return outpus as json with keys - `valid` - It's boolean, `real_data` - It's optional string or nonetype. `reason`. - String. If valid is False you need to return real fact. If it's unknown return nonetype. In any case we need reason why it's either true or no. Note: Mostly you need to analyse text and google search text because high DA/PA can lie. NOTE: if information is false, valid must uqual to False."},
                            {"role": "user", "content": evaluation_string},
                    ]
                )
                
                evaluation_output = json.loads(evaluation_request.choices[0].message.content)
                if evaluation_output['valid']:
                    return jsonify({'type': 'true', 'reason': translate_to_georgian(evaluation_output['reason']).lower()})
                else:
                    if evaluation_output['real_data']:  
                        return jsonify({'type': 'false', 'message': translate_to_georgian(evaluation_output['real_data']).lower(), 'reason': translate_to_georgian(evaluation_output['reason']).lower()})
                    else:
                        return jsonify({'type': 'true', 'reason': translate_to_georgian(evaluation_output['reason']).lower()})
            else:
                return jsonify({'status': 'error', 'message': 'error while extracting query from text'})
        if content_type == 'image':
            image_url = data['data']

            response = requests.get(image_url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                text = pytesseract.image_to_string(image)

                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Text: {text} | Based on extracted text, please return normalized text what this image is about and question. return output as json with 2 keys, `question` - It must contain question extracted from text and `information` - normalized text "},
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                            },
                        ],
                        }
                    ],
                    max_tokens=300,
                    )
    
                extracted_response = response.choices[0].message.content
                json_string = re.search(r'```json\n(.*)\n```', extracted_response, re.DOTALL).group(1)
                json_object = json.loads(json_string)

                if json_object['question']:
                    searched_data = search.run(json_object['question'])

                    evaluation_string = f"""
                                    Overview of image: {json_object['information']}
                                    Google output: {searched_data}
                                    """

                    evaluation_request = client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": f"Curren date: {datetime.datetime.now()}. You are FACT checker AI. your goal is to check if provided text from image is true or no based on various factors. You need to return outpus as json with keys - `valid` - It's boolean, `real_data` - It's optional string or nonetype. `reason`. - String. If valid is False you need to return real fact. If it's unknown return nonetype. In any case we need reason why it's either true or no. Note: Mostly you need to analyse text and google search text because high DA/PA can lie. NOTE: if information is false, valid must uqual to False"},
                            {"role": "user", "content": evaluation_string},
                        ]
                    )
                    evaluation_output = json.loads(evaluation_request.choices[0].message.content)

                    if evaluation_output['valid']:
                        return jsonify({'type': 'true', 'reason': evaluation_output['reason'].lower()})
                    else:
                        return jsonify({'type': 'false', 'message': evaluation_output['real_data'].lower(), 'reason': evaluation_output['reason'].lower()})
                else:
                    return jsonify({'status': 'error', 'message': 'error while extracting query from text'})

        
        if content_type == 'video' or content_type == 'iframe':
            video_url = data['data']
            test_dir = "tmpaudio"
            prediction_result = predict_video(video_url, test_dir)
            
            if prediction_result['classification'] == 'NDF':
               return jsonify({'type': 'true', 'reason': 'After analysing video, we determined that video does not contain Deep Fake'})
            else:
               return jsonify({'type': 'false', 'message': 'After analysing video, we determined that video contains Deep Fake', 'reason': 'Our prediction algorithm detected some artifacts'})
            return 'ok'

        if content_type == 'audio':
            audio_url = data['data']

            if '.' in audio_url:
                extension = audio_url.rsplit('.', 1)[1]
            else:
                # If the URL doesn't contain an extension, request to get content-type
                response = requests.head(audio_url)
                content_type = response.headers.get('content-type')

                content_type_to_extension = {
                    'audio/mpeg': 'mp3',
                    'audio/wav': 'wav',
                }
                extension = content_type_to_extension.get(content_type, 'unknown')

            filename = str(uuid.uuid4())
            
            target_directory = './tmpaudio'
            os.makedirs(target_directory, exist_ok=True)

            # Step 4: Download and save the audio file
            response = requests.get(audio_url)
            if response.status_code == 200:
                file_path = os.path.join(target_directory, f"{filename}.{extension}")
                with open(file_path, 'wb') as file:
                    file.write(response.content)
            else:
                print("Failed to download the audio file.")

            audio_file = open(file_path, "rb")
            text = client.audio.translations.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="text"
            )

            os.remove(file_path)

            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": f"NOTE: You need to formulate questions like `Ronaldos current football team`. Current date: {datetime.datetime.now()}. You are smart AI, your goal is to generate google search query based on chunk of text to determine this text is true or not. NOTE: formulate question to determine it's true or no. never directly use date, and text. Always formulate questions. return ouput as json"},
                    {"role": "user", "content": "Ronaldo changed club to - Barcelona"},
                    {"role": "assistant", "content": "{'query': 'Ronaldos current football team'}"},
                    {"role": "user", "content": text}
                ]
            )

            output = json.loads(response.choices[0].message.content)
            if output['query']:
                searched_data = search.run(output['query'])

                evaluation_string = f"""
                                    Text to check: {text}
                                    Google output: {searched_data}
                                    """


                evaluation_request = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": f"NOTE: YOU MUST BE VERY STRICT, IF INFORMATION IS FALSE YOU NEED TO RETURN FALSE. Curren date: {datetime.datetime.now()}. You are FACT checker AI. your goal is to check if provided text is true or no based on various factors. You need to return outpus as json with keys - `valid` - It's boolean, `real_data` - It's optional string or nonetype. `reason`. - String. If valid is False you need to return real fact. If it's unknown return nonetype. In any case we need reason why it's either true or no. Note: Mostly you need to analyse text and google search text because high DA/PA can lie"},
                        {"role": "user", "content": evaluation_string},
                    ]
                )
                evaluation_output = json.loads(evaluation_request.choices[0].message.content)

                if evaluation_output['valid']:
                    return jsonify({'type': 'true', 'reason': evaluation_output['reason']})
                else:
                    return jsonify({'type': 'false', 'message': evaluation_output['real_data'], 'reason': evaluation_output['reason']})

if __name__ == '__main__':
    app.run(debug=True)
