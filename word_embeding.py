from gensim.models import Word2Vec

senntence = [["i", "love", "learning"], ["machine", "learning", "is", "fun"]]

model = Word2Vec(senntence,vector_size = 10, window = 2, min_count = 1)

print("Vector for 'learning':", model.wv['learning'])