from nltk.tokenize import sent_tokenize, \
        word_tokenize, WordPunctTokenizer

# Define input text
input_text = "Do you know how tokenization works? It's actually quite interesting! Let's analyze a couple of sentences and figure it out." 

# Sentence tokenizer 
print("\nSentence tokenizer:")
print(sent_tokenize(input_text))

# Word tokenizer
print("\nWord tokenizer:")
print(word_tokenize(input_text))

# WordPunct tokenizer
print("\nWord punct tokenizer:")
print(WordPunctTokenizer().tokenize(input_text))
