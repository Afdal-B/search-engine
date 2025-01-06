from TD3_6 import corpus
import numpy as np
from scipy.sparse import csr_matrix
all_docs_string = " ".join([x.texte for x in corpus.id2doc.values()])

print(f"the textes : {all_docs_string}")


# découpage en mot (tri alphabétique et suppression de doublon)
all_docs_string = sorted(set(all_docs_string.lower().split()))

vocab = {}
for index, value in enumerate(all_docs_string):
    vocab[value] = {"id": index, "nbr_occur": 0}

print(f"the vocab : {vocab}")
# Initialisation des valeurs de la matrice
num_docs = len(corpus.id2doc)
num_words = len(vocab)
data = []
rows = []
cols = []

# construction de la matrice
for doc_id, doc in enumerate(corpus.id2doc.values()):
    word_counts = {}
    for word in doc.texte.lower().split():
        if word in vocab:
            word_id = vocab[word]["id"]
            if word_id not in word_counts:
                word_counts[word_id] = 0
            word_counts[word_id] += 1

    for word_id, count in word_counts.items():
        rows.append(doc_id)
        cols.append(word_id)
        data.append(count)


mat_TF = csr_matrix((data, (rows, cols)), shape=(num_docs, num_words))

print(mat_TF)

# Calcul du total d'occurance et de la fréquence
for word, info in vocab.items():
    word_id = info["id"]
    # total d'occurance du mot dans le corpus
    total_occurrences = mat_TF[:, word_id].sum()
    # Nombre de document contenant le mot
    doc_frequency = (mat_TF[:, word_id] > 0).sum()

    vocab[word]["nbr_occur"] = total_occurrences
    vocab[word]["doc_freq"] = doc_frequency

print(
    f"Mise à jour du vocabulaire avec occurrences et fréquences des documents: {vocab}"
)
print("DONE")
