from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import math
from difflib import SequenceMatcher
import json


def clean_and_tokenize(content):
    # Remove non-alphabetic characters and tokenize
    tokens = [word.lower()
              for word in word_tokenize(content) if word.isalnum()]
    # Remove stopwords
    tokens = [
        word for word in tokens if word not in stopwords.words('english')]
    return tokens


def calculate_tf(document_tokens):
    word_counts = Counter(document_tokens)
    total_words = len(document_tokens)
    tf_dict = {word: count / total_words for word,
               count in word_counts.items()}
    return tf_dict


def calculate_idf_for_documents(document_paths):
    total_documents = len(document_paths)
    word_document_count = defaultdict(int)

    for document_path in document_paths:
        with open(document_path, 'r', encoding='utf-8') as file:
            content = file.read()
            document_tokens = clean_and_tokenize(content)
            unique_words = set(document_tokens)
            for word in unique_words:
                word_document_count[word] += 1

    idf_dict = {}
    for word, count in word_document_count.items():
        idf_value = math.log((total_documents + 1) / (count + 1)) + 1
        idf_dict[word] = idf_value

    return idf_dict


def calculate_tf_idf_for_document(file_path, idf_dict):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            document_tokens = clean_and_tokenize(content)

            tf_dict = calculate_tf(document_tokens)
            tf_idf_dict = {word: tf_value *
                           idf_dict[word] for word, tf_value in tf_dict.items()}

            return tf_idf_dict

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def calculate_tf_idf_for_documents(document_paths, idf_dict):
    tf_idf_results = {}

    for document_path in document_paths:
        tf_idf_result = calculate_tf_idf_for_document(document_path, idf_dict)
        if tf_idf_result is not None:
            tf_idf_results[document_path] = tf_idf_result

    return tf_idf_results


def find_most_similar_paragraph(query, paragraphs):
    similarities = {}
    for i, paragraph in enumerate(paragraphs):
        similarity_ratio = SequenceMatcher(None, query, paragraph).ratio()
        similarities[i] = similarity_ratio

    most_similar_paragraph = max(similarities, key=similarities.get)
    return most_similar_paragraph, similarities[most_similar_paragraph]


def cosine_similarity(vector1, vector2):
    dot_product = sum(vector1[key] * vector2[key]
                      for key in set(vector1) & set(vector2))
    magnitude1 = math.sqrt(sum(value ** 2 for value in vector1.values()))
    magnitude2 = math.sqrt(sum(value ** 2 for value in vector2.values()))

    if magnitude1 * magnitude2 == 0:
        return 0  # Avoid division by zero

    return dot_product / (magnitude1 * magnitude2)


def find_most_similar_document(query, tfidf_documents, idf_dict):
    similarities = {}
    query_tokens = clean_and_tokenize(query)
    query_tfidf = {word: query_tokens.count(
        word) * idf_dict.get(word, 0) for word in query_tokens}

    for doc_path, tfidf_doc in tfidf_documents.items():
        similarity = cosine_similarity(query_tfidf, tfidf_doc)
        similarities[doc_path] = similarity

    most_similar_doc = max(similarities, key=similarities.get)
    return most_similar_doc, similarities[most_similar_doc]


selected_document_id = 22

# Load data from the JSON file
with open('D:\DS_Project (2)\data.json', 'r') as file:
    json_data = json.load(file)

# Accessing the candidate documents of "document_id": 0
document_id_n_candidates = None

for item in json_data:
    if item["document_id"] == selected_document_id:
        document_id_n_candidates = item["candidate_documents_id"]
        break

# Calculate IDF for all documents
document_paths = []

for doc_number in document_id_n_candidates:
    path = f"D:\DS_Project (2)\data\document_{doc_number}.txt"
    document_paths.append(path)

idf_documents = calculate_idf_for_documents(document_paths)

# Calculate TF-IDF for all documents
tf_idf_documents = calculate_tf_idf_for_documents(
    document_paths, idf_documents)

for item in json_data:
    if item["document_id"] == selected_document_id:
        query = item["query"]
        break

most_similar_doc, similarity_score = find_most_similar_document(
    query, tf_idf_documents, idf_documents)
with open(most_similar_doc, 'r', encoding='utf-8') as file:
    content = file.read()
    paragraphs = content.split('\n')

most_similar_paragraph, paragraph_similarity_score = find_most_similar_paragraph(
    query, paragraphs)
print(f"\nThe most similar document to the query is: {most_similar_doc}")
print(
    f"\nThe most similar paragraph to the query is: Paragraph {most_similar_paragraph + 1}")
print(f"\nSimilarity score: {similarity_score}")

selected_document_tfidf = tf_idf_documents[most_similar_doc]
important_words_tfidf = dict(
    sorted(selected_document_tfidf.items(), key=lambda x: x[1], reverse=True)[:5])
print("\nTop 5 important words (TF-IDF) in the selected document:")
print(important_words_tfidf)


selected_document_path = most_similar_doc
with open(selected_document_path, 'r', encoding='utf-8') as file:
    selected_document_content = file.read()

selected_document_tokens = clean_and_tokenize(selected_document_content)
selected_document_tf = calculate_tf(selected_document_tokens)
most_repeated_words_tf = dict(
    sorted(selected_document_tf.items(), key=lambda x: x[1], reverse=True)[:10])
print("\nTop 10 most repeated words (TF) in the selected document:")
print(most_repeated_words_tf)
