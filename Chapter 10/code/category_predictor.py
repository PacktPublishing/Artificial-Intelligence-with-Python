from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Define the category map
category_map = {'talk.politics.misc': 'Politics', 'rec.autos': 'Autos', 
        'rec.sport.hockey': 'Hockey', 'sci.electronics': 'Electronics', 
        'sci.med': 'Medicine'}

# Get the training dataset
training_data = fetch_20newsgroups(subset='train', 
        categories=category_map.keys(), shuffle=True, random_state=5)

# Build a count vectorizer and extract term counts 
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(training_data.data)
print("\nDimensions of training data:", train_tc.shape)

# Create the tf-idf transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)

# Define test data 
input_data = [
    'You need to be careful with cars when you are driving on slippery roads', 
    'A lot of devices can be operated wirelessly',
    'Players need to be careful when they are close to goal posts',
    'Political debates help us understand the perspectives of both sides'
]

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB().fit(train_tfidf, training_data.target)

# Transform input data using count vectorizer
input_tc = count_vectorizer.transform(input_data)

# Transform vectorized data using tfidf transformer
input_tfidf = tfidf.transform(input_tc)

# Predict the output categories
predictions = classifier.predict(input_tfidf)

# Print the outputs
for sent, category in zip(input_data, predictions):
    print('\nInput:', sent, '\nPredicted category:', \
            category_map[training_data.target_names[category]])

