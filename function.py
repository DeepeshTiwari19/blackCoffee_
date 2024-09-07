import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import string
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Download necessary resources
nltk.download('punkt')
nltk.download('cmudict')

def stopward_in_one_file(folder_path, output_file):
    # Set to store unique stopwords
    all_stopwords = set()

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file (and not a directory)
        if os.path.isfile(file_path):
            try:
                # Open and read the file, adding words to the set
                with open(file_path, 'r') as file:
                    stopwords = file.read().splitlines()
                    all_stopwords.update(stopwords)  # Add to the set to keep only unique words
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # Write all collected stopwords into a single 'stopwords.txt' file
    with open(output_file, 'w') as output:
        for word in sorted(all_stopwords): 
            output.write(word + '\n')

    print(f"Combined stopwords have been saved to {output_file}.")

def read_words(file_path):
    """Read words from a file and return as a set."""
    with open(file_path, 'r') as file:
        words = set(word.strip().lower() for word in file.readlines())
    return words

def compute_sentiment_scores(text, positive_words, negative_words):
    """Compute positive and negative scores based on word lists."""
    # Tokenize the text
    tokens = text.lower().split()
    
    # Count positive and negative words
    positive_score = sum(1 for word in tokens if word in positive_words)
    negative_score = sum(-1 for word in tokens if word in negative_words)
    # Convert negative score to a positive value
    negative_score = abs(negative_score)
    
    return positive_score, negative_score, tokens

def calculate_polarity_score(positive_score, negative_score):
    """Calculate Polarity Score."""
    return (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

def calculate_subjectivity_score(positive_score, negative_score, total_words):
    """Calculate Subjectivity Score."""
    return (positive_score + negative_score) / (total_words + 0.000001)

# Load CMU Pronouncing Dictionary
d = cmudict.dict()

def count_syllables(word):
    try:
        syllable_counts = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
        return syllable_counts[0] if syllable_counts else 0
    except KeyError:
        return 0
    

def calculate_average_sentence_length(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    num_sentences = len(sentences)
    num_words = len(words)
    
    average_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    average_sentence_length=round(average_sentence_length, 2)
    return average_sentence_length

def calculate_percentage_of_complex_words(text):
    words = word_tokenize(text)
    num_words = len(words)
    
    complex_word_count = sum(1 for word in words if count_syllables(word) >= 3)
    percentage_of_complex_words = (complex_word_count / num_words) * 100 if num_words > 0 else 0
    percentage_of_complex_words=round(percentage_of_complex_words, 2)
    return percentage_of_complex_words

def calculate_fog_index(text):
    average_sentence_length = calculate_average_sentence_length(text)
    percentage_of_complex_words = calculate_percentage_of_complex_words(text)
    
    fog_index = 0.4 * (average_sentence_length + percentage_of_complex_words)
    fog_index=round(fog_index, 2)
    return fog_index

def data_cleaning(texter_content):
    text = re.sub(r'\s+', ' ', texter_content)  # Replace multiple spaces with a single space
    text = re.sub(r'\[\d+\]', '', text)  # Remove references like [0-9]
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\.\s+', '. ', text)  # Fix extra spaces around periods
    text = re.sub(r'[^\w\s,.!?]', '', text)  # Remove unwanted punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove redundant spaces
    text = re.sub(r'\.\s+\.', '.', text)
    text = text.lower()  # Convert to lowercase
    print(text)
    return text

def clean_and_count_words(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove punctuation and stop words
    cleaned_words = [word.lower() for word in words if word.lower() not in stop_words and word not in punctuation]
    
    # Count the number of cleaned words
    word_count = len(cleaned_words)
    return word_count


def syllable_count_per_word(text):
    words = nltk.word_tokenize(text)
    syllable_counts = {word: count_syllables(word) for word in words}
    return syllable_counts


def count_personal_pronouns(text):
    # Define personal pronouns and a pattern to avoid 'US' as a country
    pronouns = ["I", "we", "my", "ours", "us"]
    
    # Define a regex pattern to match personal pronouns but not 'US' as a country
    # Use negative lookbehind and lookahead to ensure 'US' is not mistakenly counted
    pronoun_pattern = r'\b(?:I|we|my|ours)\b|\b(?:us)\b'
    
    # Compile the regex pattern
    pattern = re.compile(pronoun_pattern, re.IGNORECASE)
    
    # Find all matches in the text
    matches = pattern.findall(text)
    
    # Initialize a dictionary to store the counts of each pronoun
    pronoun_counts = {pronoun: 0 for pronoun in pronouns}
    
    # Count occurrences of each personal pronoun
    for match in matches:
        if match.lower() in pronoun_counts:
            pronoun_counts[match.lower()] += 1
    
    # Calculate the total count of all personal pronouns
    total_count = sum(pronoun_counts.values())
    
    return pronoun_counts, total_count

def calculate_average_word_length(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove any punctuation and non-alphabetic characters from each word
    cleaned_words = [re.sub(r'\W+', '', word) for word in words if word.isalpha()]
    
    # Calculate the total number of characters in each word
    total_characters = sum(len(word) for word in cleaned_words)
    
    # Calculate the total number of words
    total_words = len(cleaned_words)
    
    # Calculate the average word length
    average_word_length = total_characters / total_words if total_words > 0 else 0
    average_word_length=round(average_word_length, 2)
    
    return average_word_length





