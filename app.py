from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
import base64
from src.evaluate.testOneAudio import ModelForApp
import numpy as np
import subprocess
from pydub import AudioSegment


app = Flask(__name__, template_folder="static/html")

# Define the uploads folder path inside the static directory
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Add a URL rule for serving uploaded files
app.add_url_rule(
    "/static/uploads/<filename>",
    endpoint="download_file",
    build_only=True
)

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

modelObj = ModelForApp()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['GET', 'POST'])
def record():
    if request.method == 'POST':
        audio_data_str = request.form.get('audioData')
        if audio_data_str:
            audio_data = base64.b64decode(audio_data_str.split(',')[1])
            # print(f"audio in bytes: {audio_data_str.split(',')[1]}")

            ## folders and filenames 
            filename_webm = 'outfile.webm'
            filename = 'outfile.wav'
            filepath_webm = os.path.join(app.config['UPLOAD_FOLDER'],filename_webm)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save base64 decoded bytes to outfile.webm
            with open(filepath_webm, 'wb') as f:
                f.write(audio_data)

            # Convert outfile.webm to outfile.wav using FFmpeg
            subprocess.run(['ffmpeg', '-i', filepath_webm, file_path])
            subprocess.run(['rm',filepath_webm])
            prediction_result = modelObj.predict(file_path)
            target_language = request.form.get('languageSelection', '')
            return render_template('result.html', filepath=file_path, filename=filename, prediction_result=prediction_result, target_language=target_language)
    return render_template('record.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                prediction_result = modelObj.predict(file_path)
                target_language = request.form.get('languageSelection', '')
                print(f"File saved at {file_path}")
                return render_template('result.html', filepath=file_path, filename=filename, prediction_result=prediction_result, target_language=target_language)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
