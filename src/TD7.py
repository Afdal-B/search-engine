import math

import numpy as np
from scipy.sparse import csr_matrix

from model.SearchEngine import SearchEngine
from TD3_6 import load_corpus

# == Partie 1 : Construction de la matrice Documents x Mots

corpus = load_corpus()


def build_vocab(corpus):
    """Construit le vocabulaire à partir du corpus."""
    vocab = {}
    for doc in corpus.values():
        for word in doc.texte.lower().split():
            if word not in vocab:
                vocab[word] = {"id": len(vocab), "nbr_occur": 0, "doc_freq": 0}
    return vocab


def compute_tf(doc, vocab):
    """Calcule le TF (Term Frequency) pour un document."""
    tf = np.zeros(
        len(vocab)
    )  # Initialise un vecteur de zéros pour chaque mot du vocabulaire
    doc_words = (
        doc.lower().split()
    )  # Convertit le document en minuscules et le divise en mots
    for word in doc_words:  # Pour chaque mot dans le document
        if word in vocab:  # Si le mot est dans le vocabulaire
            tf[vocab[word]["id"]] += 1  # Incrémente le compteur pour ce mot
    return (
        tf / len(doc_words) if len(doc_words) > 0 else tf
    )  # Normalise en divisant par le nombre total de mots


def compute_idf(corpus, vocab):
    """Calcule l'IDF (Inverse Document Frequency) pour le corpus."""
    N = len(corpus)  # Nombre total de documents dans le corpus
    for doc in corpus.values():  # Pour chaque document dans le corpus
        doc_words = set(
            doc.texte.lower().split()
        )  # Ensemble des mots uniques du document
        for word in doc_words:  # Pour chaque mot unique
            if word in vocab:  # Si le mot est dans le vocabulaire
                vocab[word]["doc_freq"] += (
                    1  # Incrémente le nombre de documents contenant ce mot
                )

    for word, info in vocab.items():  # Pour chaque mot dans le vocabulaire
        info["idf"] = (
            math.log((N + 1) / (info["doc_freq"] + 1)) + 1
        )  # Calcule l'IDF avec lissage


def compute_tfidf(corpus, vocab):
    """Calcule la matrice TF-IDF pour le corpus."""
    N = len(corpus)  # Nombre de documents
    M = len(vocab)  # Taille du vocabulaire
    tfidf_matrix = np.zeros((N, M))  # Initialise la matrice TF-IDF avec des zéros

    for i, doc in enumerate(corpus.values()):  # Pour chaque document dans le corpus
        tf = compute_tf(doc.texte, vocab)  # Calcule le TF pour ce document
        for word, info in vocab.items():  # Pour chaque mot dans le vocabulaire
            j = info["id"]  # Récupère l'ID du mot
            tfidf_matrix[i, j] = tf[j] * info["idf"]  # Calcule le score TF-IDF

    return csr_matrix(tfidf_matrix)  # Retourne la matrice sous forme sparse


# Construction du vocabulaire
vocab = build_vocab(corpus.id2doc)

# Calcul de l'IDF
compute_idf(corpus.id2doc, vocab)

# Calcul de la matrice TF-IDF
mat_TF_IDF = compute_tfidf(corpus.id2doc, vocab)

print(f"Matrice TF-IDF: {mat_TF_IDF.data}")

# == Partie 2 : Moteur de recherche


def cosine_similarity(vec1, vec2):
    """Calcule la similarité cosinus entre un vecteur dense et une matrice creuse."""
    vec1 = np.ravel(vec1)
    vec2 = np.ravel(vec2)
    dot_product = vec2.dot(vec1)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 != 0:
        return dot_product / (norm1 * norm2)
    return 0


# Demander à l'utilisateur d'entrer des mots-clés
query = input("Entrez quelques mots-clefs: ").lower().split()

# Transformer les mots-clés en vecteur TF-IDF
query_vector = np.zeros((1, len(vocab)))
for word in query:
    if word in vocab:
        word_id = vocab[word]["id"]
        query_vector[0, word_id] += 1

# Calculer la similarité cosinus entre la requête et tous les documents
similarities = np.zeros((mat_TF_IDF.shape[0], 1))
for i in range(mat_TF_IDF.shape[0]):
    similarities[i] = cosine_similarity(
        query_vector, np.ravel((mat_TF_IDF.toarray())[i])
    )

# Aplatir pour obtenir un vecteur 1D
flat_array = similarities.flatten()

# Supprimer les NaN (ou les gérer selon le besoin)
clean_array = flat_array[~np.isnan(flat_array)]

# Obtenir les indices des valeurs triées par ordre décroissant
sorted_indices = np.argsort(clean_array)[::-1]

print("Valeurs triées par ordre décroissant :")
top_k = 5
for idx in sorted_indices[:top_k]:
    print({"Document ID": idx, "Similarity": clean_array[idx]})
    print("D++++++", corpus.id2doc[idx])


# === Exemple d'utilisation avec la classe SearchEngine
print("\n ============> Utilisation de la classe SearchEngine:")
search_engine = SearchEngine(corpus)
query = input("Entrez quelques mots-clefs: ")
results = search_engine.search(query)
print("Meilleurs résultats :")
print(results)
