# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 00:55:34 2020

@author: ali_e

"""
import numpy as np
import pandas as pd

#reads in compressed .json.gz file to pandas dataframe
def createPdDF(path):
    '''
    Enter the path name to the .json.gz file as string
    Pandas DataFrame will be returned
    '''
    return pd.read_json(path, compression='gzip', lines=True)



def cleanDF(dataframe):
    '''
    Pass in a dataframe that contains an amazon review dataset to be cleaned.
    Assumption: dataframe was loaded in and no manipulation has occured.
    
    Returns cleaned dataframe
    '''
    #drop 'style', 'image' and 'unixReviewTime' columns
    modified_df = dataframe.drop(['style','image', 'unixReviewTime'], axis = 1)
    
    #remove rows with no review text
    modified_df = modified_df.dropna(axis = 0, subset = ['reviewText'])
    
    #first need to remove commas from vote column in order to convert to type int
    modified_df['vote'] = modified_df['vote'].str.replace(',','')
    #fill NaNs in vote column with zeroes and change votes from type object to int
    modified_df['vote'] = modified_df['vote'].fillna(0).astype(int)
    
    #fill empty summaries with review text
    modified_df['summary'].fillna(modified_df['reviewText'], inplace = True)
    
    #fill empty names with 'Amazon Customer'
    modified_df['reviewerName'].fillna('Amazon Customer', inplace = True)
    
    #change review time to type datetime for later modifications
    modified_df = modified_df.astype({'reviewTime': 'datetime64[ns]'})
    
    #Convert True and False under verified purchase column to binary.
    modified_df['verified'] = modified_df['verified'].astype(int)

    #remove duplicates
    modified_df.drop_duplicates(inplace=True)
    
    #need to reset index
    modified_df = modified_df.reset_index().drop('index', axis=1)

    return modified_df



def featureEngin(dataframe):
    '''
    This functions create several new features and adds them to the dataframe that is passed through as the argument.
    
    New features:
    
    1. Adds review word count and summary word count to the DataFrame.
    
    2. Extracts the cyclical components of the date and creates a column for each.
    The cyclical componenets are month(1-12) and day of week(0-6).  
    
    3. Creating feature that indicates whether each review is the only review a reviewer has posted or if they have posted other reviews as well (no=0, yes=1)
    
    4. Create new features that display if a reviewer only gave 5 star reviews or one star reviews (both are binary features)
    
    5. Get number of reviews associated with each product (asin) to be used as a popularity index

    6. Checking if customer provided their name or not (binary)

    Returns original dataframe that was passed in with these new features listed above.
    '''
    #adding word counts for review and summary columns
    dataframe['review_word_count'] = dataframe['reviewText'].str.split().str.len()
    dataframe['summary_word_count'] = dataframe['summary'].str.split().str.len()
    
    #get the month the review was posted in
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html
    dataframe['month'] = pd.DatetimeIndex(dataframe['reviewTime']).month

    #get the day of the week the review was posted as an integer (Monday=0,Sunday=6)
    dataframe['dayofweek'] = pd.DatetimeIndex(dataframe['reviewTime']).dayofweek
    
    #creating feature that indicates whether each review is the only review a reviewer has posted or 
    #if they have posted other reviews as well (no=0, yes=1)
    #1. start by getting a dictionary of the number of reviews per unique reviewer
    map_numreviews = dataframe['reviewerID'].value_counts().to_dict()
    #2. map the dictionary above to the reviewers in the dataframe and create a new column from this
    dataframe['multipleReviews_reviewer'] = dataframe['reviewerID'].map(map_numreviews)
    #3. finally, convert the number of reviews to binary: 1 review = 0; more than 1 review = 1
    dataframe['multipleReviews_reviewer'] = np.where(dataframe['multipleReviews_reviewer'] > 1, 1, 0)
    
    #create new features that display if a reviewer only gave 5 star reviews or one star reviews
    #1. need a dictionary of booleans which checks if they only have 5 or 1 star reviews
    map_five = dataframe['overall'].groupby(dataframe['reviewerID']).agg(lambda x: (np.unique(x)==5).all()).to_dict()
    map_one = dataframe['overall'].groupby(dataframe['reviewerID']).agg(lambda x: (np.unique(x)==1).all()).to_dict()
    #2. create new columns with mappings created above
    dataframe['reviewer_five_star_only'] = dataframe['reviewerID'].map(map_five)
    dataframe['reviewer_one_star_only'] = dataframe['reviewerID'].map(map_one)
    #3. convert booleans to binary
    dataframe['reviewer_five_star_only'] = dataframe['reviewer_five_star_only'].astype(int)
    dataframe['reviewer_one_star_only'] = dataframe['reviewer_one_star_only'].astype(int)

    #get number of reviews associated with each product (asin) - to be used as a popularity index
    #1. create a map of the number of reviews per product
    map_numreviews = dataframe['asin'].value_counts().to_dict()
    #2. map the dictionary to the asin (product IDs) and create a new column based off that
    dataframe['numReviews_product'] = dataframe['asin'].map(map_numreviews) 
    
    #checking if customer provided their name or not (binary)
    dataframe['nameProvided'] = np.where(dataframe['reviewerName'] != 'Amazon Customer',1,0)

    return dataframe



def pipedf(path):
    '''
    Calls all the functions:
    - createPdDF: import datafile (in the form of json.gz) and convert it to pd DataFrame
    - cleanDF: clean the dataframe
    - featureEngin: create new numeric features
    Returns a pd DataFrame with the above functions applied.
    '''
    return featureEngin(cleanDF(createPdDF(path)))



# Processing Data functions

#importing relevant libraries
from nltk.corpus import stopwords 
ENGLISH_STOP_WORDS = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def spl_tokenizer(sentence):
    '''
    Tokenizer with following specs:
    - removes english stopwords
    - removes punctuation
    - lemmatizes words
    '''
    for punctuation_mark in string.punctuation:
        # Remove punctuation and set to lower case
        sentence = sentence.replace(punctuation_mark,'').lower()

    # split sentence into words
    listofwords = sentence.split(' ')
    listoflemmatized_words = []
        
    # Remove stopwords and any tokens that are just empty strings
    for word in listofwords:
        if (not word in ENGLISH_STOP_WORDS) and (word!=''):
            # Lemmatize words
            token = WordNetLemmatizer().lemmatize(word)            
            
            #add r_ prefix to indicate word is from review or s_ if from summary
            try:
                if tfidf.type == 'review':
                    token = 'r_' + token
                elif tfidf.type == 'summary':
                    token = 's_' + token
            except:
                pass

            #append token to list
            listoflemmatized_words.append(token)

    return listoflemmatized_words



def sps_tokenizer(sentence):
    '''
    Tokenizer with following specs:
    - removes english stopwords
    - removes punctuation
    - stems words
    '''
    for punctuation_mark in string.punctuation:
        # Remove punctuation and set to lower case
        sentence = sentence.replace(punctuation_mark,'').lower()

    # split sentence into words
    listofwords = sentence.split(' ')
    listofstemmed_words = []
    #instantiate stemmer
    stemmer = PorterStemmer() 

    # Remove stopwords and any tokens that are just empty strings
    for word in listofwords:
        if (not word in ENGLISH_STOP_WORDS) and (word!=''):
            # Lemmatize words
            token = stemmer.stem(word)  

            #add r_ prefix to indicate word is from review or s_ if from summary
            try:
                if tfidf.type == 'review':
                    token = 'r_' + token
                elif tfidf.type == 'summary':
                    token = 's_' + token
            except:
                continue

            #append token to list
            listofstemmed_words.append(token)

    return listofstemmed_words




def tfidf(dataframe_column, tokenizer, min_df=0.02, max_df=0.8, ngram_range=(1,1)):

    #0. For tokenization, need to determine what prefix (r_ or s_) to add to each token
    #To do so, retrieve the name of the column from which the tokens are generated
    column_name = dataframe_column.name
    #Next, assign an attribute (called type) to the tfidf function to indicate if the tokens are from review or summary
    if column_name == 'reviewText':
        tfidf.type = 'review'
    elif column_name == 'summary':
        tfidf.type = 'summary'
    else:
        tfidf.type = 'none'

    # 1. Instantiate  (stop_words='english')
    vectorizer = TfidfVectorizer(min_df = min_df, max_df = max_df, tokenizer = tokenizer, ngram_range = ngram_range)
    
    # 2. Fit 
    vectorizer.fit(dataframe_column)
    
    # 3. Transform
    reviews_tokenized = vectorizer.transform(dataframe_column)
    
    # We extract the information and put it in a data frame
    tokens = pd.DataFrame(columns=vectorizer.get_feature_names(), data=reviews_tokenized.toarray())
    
    return tokens

