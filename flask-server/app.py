from flask import Flask, request, jsonify
import subprocess
import csv
from flask_cors import CORS
from functions import process_audio, search_movie, predict_genre, xai_and_predict
from distilFunction import classify_text

app = Flask(__name__)
CORS(app)

@app.route('/process_audio', methods=['POST'])
def process_audio_route():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']

    temp_file = 'temp_audio.webm'
    audio_file.save(temp_file)

    converted_file = 'temp_audio.wav'
    subprocess.run(['ffmpeg', '-y', '-i', temp_file, converted_file])

    recognized_text = process_audio(converted_file)

    csv_file = '/Users/sandundesilva/Documents/4th year/Research Project/UI/findMyFilm/flask-server/Movie Final Filtered - movies3.csv'
    movie_name = search_movie(csv_file, recognized_text)

    return jsonify({'recognized_text': recognized_text, 'movie_name': movie_name})

@app.route('/text-submit', methods=['POST'])
def text_submit_route():
    if request.is_json:
        data = request.get_json()
        recognized_text = data.get('textInput')
        if recognized_text:
            csv_file = '/Users/sandundesilva/Documents/4th year/Research Project/UI/findMyFilm/flask-server/Movie Final Filtered - movies3.csv'
            movie_name = search_movie(csv_file, recognized_text)
            return jsonify({'movie_name': movie_name})
        else:
            return jsonify({'error': 'No textInput provided'}), 400
    else:
        return jsonify({'error': 'Request data must be JSON'}), 415
    

@app.route('/predict-genre', methods=['POST'])
def predicted_genre_route():
    if request.is_json:
        data = request.get_json()
        best_movie = data.get('movieTitle')
        if best_movie:
            csv_file = '/Users/sandundesilva/Documents/4th year/Research Project/UI/findMyFilm/flask-server/Movie Final Filtered - movies3.csv'
            predict_genres = predict_genre(csv_file,best_movie)
            return jsonify({"movie_genre": predict_genres})
        else:
            return jsonify({'error': 'No movieTitle provided'}), 400
    else:
        return jsonify({'error': 'Request data must be JSON'}), 415

@app.route('/predict', methods=['POST'])
def predict():
    subtitle = request.form['subtitle']
    predicted_genre, attributions, convergence_delta = xai_and_predict(subtitle)
    return jsonify({
        'predicted_genre': predicted_genre,
        'attributions': attributions,
        'convergence_delta': convergence_delta
    })

@app.route('/predict-distil', methods=['POST'])
def predict_distil():
    subtitle = request.form['subtitle']
    predicted_genre = classify_text(subtitle)
    return jsonify({
        'predicted_genre': predicted_genre,
    })

@app.route('/home')
def home():
    return "Welcome to the home page"

if __name__ == '__main__':
    app.run(debug=True)
