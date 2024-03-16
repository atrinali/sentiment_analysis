import spacy
import pandas as pd

nlp = spacy.load('en_core_web_md') # loading medium model

amazon_df = pd.read_csv('amazon_product_reviews.csv')

# selecting first 100 rows from column 'reviews.text', using double brackets to extract as dataframe
reviews_data = amazon_df[['reviews.text']].head(500)

reviews_data.isnull().sum() # checking for null values
# function to extract tokens through lemmatization, and removing stop words and punctuation
def preprocess(text):
    # converting text to lowercase, removing leading/ending whitespaces and putting through spacy NLP pipeline
    doc = nlp(text.lower().strip())
    # extracting tokens through lemmatization, and removing stop words, punctuation, numbers, single letters, and special characters
    processed = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.text.isdigit() and len(token.text) > 1 and token.text.isalpha()]

    # joining token words with ' ' separator
    return ' '.join(processed)

# appyling preprocess function and adding column with processed reviews text
reviews_data['processed.text'] = reviews_data['reviews.text'].apply(preprocess)

#reviews_data.head()
from textblob import TextBlob # importing class used for processing textual data

# Initializing positive and negative counts for whole review
positive_count = 0
negative_count = 0

# Iterating through processed text
for text in reviews_data['processed.text']:
    # Creating TextBlob object for the processed text
    review_blob = TextBlob(text)
    
    # Calculating polarity for the entire review
    polarity = review_blob.sentiment.polarity

    # Updating counts in dictionaries based on polarity for current review
    if polarity > 0:
        positive_count += 1
    elif polarity < 0:
        negative_count += 1

print("Counting number of positive and negative reviews in the Data Set:")
print("Number of positive reviews: ", positive_count,
    "\nNumber of negative reviews: ", negative_count)

# comparing the similarity of two product reviews
print("\nLet's calculate similarity score for two random reviews:")
my_review_of_choice1 = nlp(reviews_data['reviews.text'][202])
my_review_of_choice2 = nlp(reviews_data['reviews.text'][59])

# calculating the similarity of two chosen product reviews
similarity = nlp(my_review_of_choice1).similarity(my_review_of_choice2)

print("1) ", my_review_of_choice1)
print("2) ", my_review_of_choice2)
print("Similarity score for sentences: ", similarity)
