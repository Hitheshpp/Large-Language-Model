import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

embeddings = {}
with open("glove.6B.50d.txt","r",encoding = "utf-8") as f:
    for line in f:
        value = line.split()
        word = value[0]
        vector = np.array(value[1:],dtype = "float32")
        embeddings[word] = vector

# Preprocessing function (clean text, tokenize, vectorize)
def preprocess_and_vectorize(text, embeddings):
    tokens = text.lower().split()
    vectors = [embeddings[token] for token in tokens if token in embeddings]
    if len(vectors) == 0:
        return np.zeros(50)  # Handle unknown words
    return np.mean(vectors, axis=0)

# Dataset: Sample sentences and labels
data = [
    ("I love this movie!", "Positive"),
    ("This is a bad product.", "Negative"),
    ("Amazing experience!", "Positive"),
    ("I hate it.", "Negative"),
]

# Prepare data
texts, labels = zip(*data)
X = np.array([preprocess_and_vectorize(text, embeddings) for text in texts])
y = np.array([1 if label == "Positive" else 0 for label in labels])  # 1 for positive, 0 for negative

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test the model
sample_text = "This is a bad product."
vector = preprocess_and_vectorize(sample_text, embeddings)
print("Sentiment:", "Positive" if model.predict([vector])[0] == 1 else "Negative")