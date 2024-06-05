import csv
import re
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio
import torch
from thefuzz import fuzz, process
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

model_id = "/Users/sandundesilva/Documents/4th year/Research Project/UI/findMyFilm/flask-server/Models/final/whisper_model"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Define genre labels
genre_labels = {
    0: 'Action',
    1: 'Adventure',
    2: 'Crime',
    3: 'Fantasy',
    4: 'Family',
    5: 'Horror',
    6: 'mystery',
    7: 'Romance',
    8: 'Sci-Fi',
    9: 'Thriller'
}

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

def process_audio(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    sample = {"raw": waveform[0].numpy(), "sampling_rate": sample_rate}

    result = pipe(sample)

    recognized_text = result["text"]
    print("Recognized Text:", recognized_text)

    return recognized_text

def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

# def search_movie(csv_file, recognized_text):
#     cleaned_recognized_text = clean_text(recognized_text)
#     with open(csv_file, newline='', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             cleaned_paragraph = clean_text(row['Paragraph'])
#             if cleaned_recognized_text.lower() in cleaned_paragraph.lower():
#                 cleaned_movie_name = clean_text(row['Movie'])
#                 original_movie_name = row['Movie']
#                 print("Found movie:", original_movie_name)
#                 return original_movie_name
#     print("Movie not found")
#     return "Couldn’t quite catch that"

def search_movie(csv_file, recognized_text):
    cleaned_recognized_text = clean_text(recognized_text)
    max_score = 0
    best_movie = "Couldn’t quite catch that"
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cleaned_subtitle = clean_text(row['Paragraph'])
            score = fuzz.partial_ratio(cleaned_recognized_text, cleaned_subtitle)
            if score > max_score:
                max_score = score
                best_movie = row['Movie']
    if max_score >= 40:
        print(f"Found movie: {best_movie} with a score of {max_score}.")
    else:
        print("Movie not found.")
        best_movie = "Couldn’t quite catch that"
    return best_movie

movie_data = pd.read_csv('/Users/sandundesilva/Documents/4th year/Research Project/UI/findMyFilm/flask-server/Models/test/Movie Final Filtered - movies3.csv')

def find_paragraph_for_movie(movie_name):
    movie_row = movie_data[movie_data['Movie'] == movie_name]
    if not movie_row.empty:
        return movie_row['Paragraph'].values[0]
    else:
        return None

def split_paragraph_into_sentences(paragraph):
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    return sentences

def split_into_nearest_sentences(paragraph, max_words=200):
    sentences = split_paragraph_into_sentences(paragraph)
    result = []
    current_sentence = ""

    for sentence in sentences:
        words = sentence.split()
        if len(current_sentence.split()) + len(words) <= max_words:
            current_sentence += " " + sentence.strip()
        else:
            result.append(current_sentence.strip())
            current_sentence = sentence.strip()

    if current_sentence:
        result.append(current_sentence)

    return result

tokenizer = AutoTokenizer.from_pretrained("shaggysus/MovieGenrePrediction")
model = AutoModelForSequenceClassification.from_pretrained("shaggysus/MovieGenrePrediction")


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def find_most_similar_movie(input_text, csv_file):
    max_similarity = 0
    most_similar_movie = None
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            movie_name = row['Movie']
            paragraph = row['Paragraph']
            similarity = 1 - (levenshtein_distance(input_text, paragraph) / max(len(input_text), len(paragraph)))
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_movie = movie_name
                
    return most_similar_movie, max_similarity


def predict_genre_for_paragraph(paragraph, window_size=512, overlap=50):
    segments = []
    start = 0
    while start < len(paragraph):
        end = min(start + window_size, len(paragraph))
        segments.append(paragraph[start:end])
        start += window_size - overlap

    genre_predictions = []
    for segment in segments:
        inputs = tokenizer(segment, return_tensors="pt", truncation=True, max_length=window_size)
        outputs = model(**inputs)
        probabilities = outputs.logits.softmax(dim=1)
        config = model.config
        class_names = config.id2label
        genre_probabilities = []
        for idx, genre in class_names.items():
            probability = probabilities[0][idx].item()
            genre_probabilities.append((genre, probability))
        genre_probabilities.sort(key=lambda x: x[1], reverse=True)
        genre_predictions.append(genre_probabilities[0])

    return genre_predictions

def predict_genre_for_paragraph_with_scene_numbers(movie_name):
    paragraph = find_paragraph_for_movie(movie_name)

    if paragraph:
        sentences = split_into_nearest_sentences(paragraph)
        genre_predictions = predict_genre_for_paragraph(paragraph)
        print("Predicted genres for each sentence:")
        for i, (sentence, prediction) in enumerate(zip(sentences, genre_predictions)):
            print(f"Scene {i+1}: Predicted Genre - {prediction[0]}, Accuracy - {prediction[1]:.4f}")
        return genre_predictions


    else:
        print("Movie not found in the dataset.")

def predict_genre(csv_file, best_movie,):
    movie_subtitle = None
    predict_genres = None
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Movie'] == best_movie:
                movie_subtitle = row['Paragraph']
                break
    
    if movie_subtitle:
        predict_genres = predict_genre_for_paragraph_with_scene_numbers(best_movie)
    else:
        print(f"Subtitle not found for movie '{best_movie}'.")
    
    return predict_genres

def predict_gen(input_ids, attention_mask=None):
    output = model(input_ids, attention_mask=attention_mask)
    return output.logits

def xai_and_predict(subtitle):
    inputs = tokenizer(subtitle, return_tensors="pt", truncation=True, padding=True)
    inputs.to(device)

    genre_logits = predict_gen(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    predicted_genre_id = torch.argmax(genre_logits, dim=1).item()

    target_index = torch.argmax(genre_logits, dim=1)
    lig = LayerIntegratedGradients(predict_gen, model.distilbert.embeddings) 
    attributions, delta = lig.attribute(inputs['input_ids'], target=target_index, return_convergence_delta=True)
    attributions = attributions.sum(dim=-1).squeeze(0)

    return genre_labels.get(predicted_genre_id, 'Unknown'), attributions.tolist(), delta.item()
