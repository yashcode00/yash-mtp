from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
import base64
import wave
from src.evaluate.testOneAudio import ModelForApp
import numpy as np

app = Flask(__name__, template_folder="static/html/")
modelObj = ModelForApp()

@app.route('/')
def index():
    return render_template('index.html')

def save_wav(audio_data, file_path, sample_rate=44100, bit_depth=16):
    """
    Save audio data as a WAV file.
    """
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(bit_depth // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

def preprocess_audio(audio_data):
    """
    Preprocess audio data to reduce noise and normalize levels.
    """
    # Apply noise reduction or filtering techniques here if available
    # Normalize audio levels
    normalized_audio = audio_data / np.max(np.abs(audio_data)) * (2 ** 15 - 1)
    return normalized_audio.astype(np.int16)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audioData' in request.form:
        audio_data_str = request.form['audioData']
        if ',' in audio_data_str:
            audio_data = base64.b64decode(audio_data_str.split(',')[1])
            # Preprocess audio
            processed_audio = preprocess_audio(audio_data)
            filename = 'recorded_audio.wav'
            file_path = os.path.join('uploads', filename)
            # Save processed audio as WAV file
            save_wav(processed_audio, file_path)
            prediction_result = modelObj.predict(file_path)
            target_language = request.form.get('languageSelection', '')
            return render_template('result.html', filename=filename, prediction_result=prediction_result, target_language=target_language)

    elif 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            prediction_result = modelObj.predict(file_path)
            target_language = request.form.get('languageSelection', '')
            return render_template('result.html', filename=filename, prediction_result=prediction_result, target_language=target_language)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
