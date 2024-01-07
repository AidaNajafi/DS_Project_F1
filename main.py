
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
import math
from difflib import SequenceMatcher
import json
from nltk.corpus import stopwords



def clean_and_tokenize(content):
    # Remove non-alphabetic characters and tokenize
    tokens = [word.lower()
              for word in word_tokenize(content) if word.isalnum()]
    # Remove stopwords
    tokens = [
        word for word in tokens if word not in stopwords.words('english')]
    return tokens

def calculate_tf(paragraph):
    word_counts = Counter(paragraph)
    total_words = len(paragraph)
    tf_dict = {word: count / total_words for word, count in word_counts.items()}
    return tf_dict

# def calculate_idf(paragraph_tokens):
#     total_paragraphs = len(paragraph_tokens)
#     word_paragraph_count = defaultdict(int)

#     for paragraph in paragraph_tokens:
#         unique_words = set(paragraph)
#         for word in unique_words:
#             word_paragraph_count[word] += 1

def calculate_idf(document_tokens):
    total_documents = 1  # Since we're dealing with a single document
    word_document_count = defaultdict(int)

    unique_words = set(document_tokens)
    for word in unique_words:
        word_document_count[word] += 1

    idf_dict = {}
    for word, count in word_document_count.items():
        idf_value = math.log((total_documents + 1) /
                             (count + 1)) + 1  # Laplace smoothing
        idf_dict[word] = idf_value

    return idf_dict

    # idf_dict = {word: math.log10(total_paragraphs / (count + 1)) for word, count in word_paragraph_count.items()}
    # return idf_dict

def calculate_tf_idf(tf_dict, idf_dict):
    tf_idf_dict = {word: abs(tf) * idf_dict[word] for word, tf in tf_dict.items()}
    return tf_idf_dict

def calculate_tf_idf_for_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the content of the file
            content = file.read()

            # Split the content into paragraphs (assuming paragraphs are separated by two newline characters)
            paragraphs = [paragraph.strip() for paragraph in content.split('\n\n')]

            # Tokenize each paragraph
            paragraph_tokens = [word_tokenize(paragraph.lower()) for paragraph in paragraphs]

            # Calculate TF for each paragraph
            tf_for_paragraphs = {}
            for i, paragraph in enumerate(paragraph_tokens, start=1):
                tf_dict = calculate_tf(paragraph)
                tf_for_paragraphs[f'Paragraph {i}'] = tf_dict

            # Calculate IDF for each word
            idf_dict = calculate_idf(paragraph_tokens)

            # Calculate TF-IDF for each word in each paragraph
            tf_idf_for_document = {}
            for paragraph_num, tf_dict in tf_for_paragraphs.items():
                tf_idf_dict = calculate_tf_idf(tf_dict, idf_dict)
                tf_idf_for_document[paragraph_num] = tf_idf_dict

            return tf_idf_for_document

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_tf_idf_for_documents():
    try:
        # Prompt the user to enter the number of documents
        num_documents = int(input("Enter the number of documents: "))

        document_paths = []
        for i in range(num_documents):
            path = input(f"Enter the path for document {i + 1}: ")
            document_paths.append(path)

        tf_idf_results = {}

        for document_path in document_paths:
            tf_idf_result = calculate_tf_idf_for_document(document_path)
            if tf_idf_result is not None:
                tf_idf_results[document_path] = tf_idf_result

        return tf_idf_results

    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return None

# Example usage
tf_idf_results = calculate_tf_idf_for_documents()

if tf_idf_results is not None:
    # Print the TF-IDF for each document
    for document_path, tf_idf_result in tf_idf_results.items():
        print(f"\nTF-IDF for {document_path}:")
        for paragraph_num, tf_idf_dict in tf_idf_result.items():
            print(f"  Paragraph {paragraph_num}: {tf_idf_dict}")







