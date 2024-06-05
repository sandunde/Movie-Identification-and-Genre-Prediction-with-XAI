from flask import Flask, request, jsonify
import subprocess
import torchaudio
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/process_audio": {"origins": "http://localhost:3000/"}})


app = Flask(__name__)


model_id = "/Users/sandundesilva/Documents/4th year/Research Project/UI/movie-ui/src/Models/final/whisper_model"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)


pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

@app.route('/process_audio', methods=['POST'])
def process_audio():

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']


    temp_file = 'temp_audio.webm'
    audio_file.save(temp_file)


    converted_file = 'temp_audio.wav'
    subprocess.run(['ffmpeg', '-y', '-i', temp_file, converted_file])


    waveform, sample_rate = torchaudio.load(converted_file)
    sample = {"raw": waveform[0].numpy(), "sampling_rate": sample_rate}


    result = pipe(sample)

    recognized_text = result["text"]
    print("Recognized Text:", recognized_text)


    return jsonify({'recognized_text': recognized_text})

if __name__ == '__main__':
    app.run(debug=True) 
