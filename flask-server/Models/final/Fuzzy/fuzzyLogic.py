import re

def preprocess_sentence(sentence):
    sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence.lower())
    return sentence

def fuzzy_sentence_similarity(sentence1, sentence2):
    sentence1 = preprocess_sentence(sentence1)
    sentence2 = preprocess_sentence(sentence2)

    words1 = sentence1.split()
    words2 = sentence2.split()

    # check how many commom wprds are there in the sentence
    common_words = set(words1) & set(words2)
    num_common_words = len(common_words)

    # Calculating the word similarity considering word order
    total_similarity = sum(1 for word1, word2 in zip(words1, words2) if word1 == word2)

    max_length = max(len(words1), len(words2))
    similarity_ratio = (num_common_words + total_similarity) / (2 * max_length)

    return min(1.0, similarity_ratio)

audio_sentence = "I love this movie alot"
reference_sentence = "The love the movie alot"

similarity = fuzzy_sentence_similarity(audio_sentence, reference_sentence)
print("Similarity between the sentences:", similarity)
