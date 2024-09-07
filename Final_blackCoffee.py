from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import nltk
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import time
from function import stopward_in_one_file, read_words,compute_sentiment_scores,calculate_polarity_score,calculate_subjectivity_score,calculate_average_sentence_length,calculate_percentage_of_complex_words,calculate_fog_index,count_syllables,data_cleaning,clean_and_count_words,syllable_count_per_word,count_personal_pronouns,calculate_average_word_length

# Download necessary resources
nltk.download('punkt')
nltk.download('cmudict')
nltk.download('stopwords')

data=pd.read_excel('C:\\Users\\dell\\Downloads\\Input.xlsx')
data.head()


# Define folder path and output file
Stopward_folder_path = '.\\StopWords'
# File paths
positive_file = '.\\negative-words.txt'
negative_file = '.\\positive-words.txt'
output_file = 'stopwords.txt'

# Call the function to collect and save stopwords
stopward_in_one_file(Stopward_folder_path, output_file)

df = pd.DataFrame(columns=['URL_ID','URL','POSITIVE SCORE', 'NEGATIVE SCORE','POLARITY SCORE','SUBJECTIVITY SCORE','AVG SENTENCE LENGTH','PERCENTAGE OF COMPLEX WORDS','FOG INDEX','AVG NUMBER OF WORDS PER SENTENCE','COMPLEX WORD COUNT','WORD COUNT','SYLLABLE PER WORD','PERSONAL PRONOUNS','AVG WORD LENGTH'])


for i in range(len(data['URL'])):
    URL_ID= data["URL_ID"][i]
    URL= data["URL"][i]
    service = Service()
    options = webdriver.ChromeOptions()
    options.add_argument("start-maximized")
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(data['URL'][i])
    time.sleep(3)
    texter = driver.find_element('xpath', './/div[@class="td-pb-span8 td-main-content"]//div[@class="td-ss-main-content"]//div[@class="td-post-content tagdiv-type"]')
    alltext=texter.text
    print(alltext)
    cleanText=data_cleaning(alltext)
    sentences = nltk.sent_tokenize(cleanText)
    with open('stopwords.txt', 'r') as file:
        stopwords = file.read().splitlines()
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [word for word in words if word not in stopwords]
        sentences[i] = ' '.join(words)
    text = ' '.join(sentences)
    print("1")


    # Read positive and negative words from files
    positive_words = read_words(positive_file)
    negative_words = read_words(negative_file)

    # Compute sentiment scores
    positive_score, negative_score, tokens = compute_sentiment_scores(text, positive_words, negative_words)

    # Calculate Polarity Score
    polarity_score = calculate_polarity_score(positive_score, negative_score)

    # Calculate Subjectivity Score
    subjectivity_score = calculate_subjectivity_score(positive_score, negative_score, len(tokens))

    # Calculate metrics
    average_sentence_length = calculate_average_sentence_length(text)
    percentage_of_complex_words = calculate_percentage_of_complex_words(text)
    fog_index = calculate_fog_index(text)

    # Calculate Complex Word Count
    complex_word_count = sum(1 for word in word_tokenize(text) if count_syllables(word) > 2)

    # Calculate clean Word Count
    word_count = clean_and_count_words(text)

    syllable_counts = syllable_count_per_word(text)
    # Count occurrences of each syllable count
    syllable_count_frequency = Counter(syllable_counts.values())

    # Find the most common syllable count
    most_common_syllable_count, _ = syllable_count_frequency.most_common(1)[0]
    # Find all words with the most common syllable count
    most_common_syllable_words = [word for word, count in syllable_counts.items() if count == most_common_syllable_count]

    # Calculate personal pronoun counts and average word length
    pronoun_counts, total_count = count_personal_pronouns(text)
    average_word_length = calculate_average_word_length(text)

    df = df._append(pd.DataFrame({'URL_ID':URL_ID,
                                  'URL':URL,
                                'POSITIVE SCORE': positive_score,
                                'NEGATIVE SCORE': negative_score,
                                'POLARITY SCORE':polarity_score,
                                'SUBJECTIVITY SCORE':subjectivity_score,
                               'AVG SENTENCE LENGTH':average_sentence_length,
                               'PERCENTAGE OF COMPLEX WORDS':percentage_of_complex_words,
                               'FOG INDEX':fog_index,
                              'AVG NUMBER OF WORDS PER SENTENCE':average_sentence_length,
                               'COMPLEX WORD COUNT':complex_word_count,
                               'WORD COUNT':word_count,
                               'SYLLABLE PER WORD':most_common_syllable_count,
                               'PERSONAL PRONOUNS':total_count,
                               'AVG WORD LENGTH':average_word_length},index=[0]), ignore_index=True)
    
    
    driver.quit()
df.to_excel('Output Data Structure.xlsx', sheet_name='Sheet1', index=False)





