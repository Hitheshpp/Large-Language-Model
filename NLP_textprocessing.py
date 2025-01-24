import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

text = "I love learning about artificial intelligence and machine learning!"

tokens = word_tokenize(text)

tokens = [word.lower() for word in tokens]

stop_words = set(stopwords.words('english'))
filtered_tocken = [word for word in tokens if word not in stop_words]

stammer = PorterStemmer()
stammered_tocken = [stammer.stem(word) for word in filtered_tocken]

print("Original Text:", text)
print("Tokens:", tokens)
print("Filtered Tokens:", filtered_tocken)
print("Stemmed Tokens:", stammered_tocken)