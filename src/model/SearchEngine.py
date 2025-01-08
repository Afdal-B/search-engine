import math

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class SearchEngine:
    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab = self.build_vocab(corpus.id2doc)
        self.compute_idf(corpus.id2doc, self.vocab)
        self.mat_TF_IDF = self.compute_tfidf(corpus.id2doc, self.vocab)

    def build_vocab(self, corpus):
        """Construit le vocabulaire à partir du corpus."""
        vocab = {}
        for doc in corpus.values():
            for word in doc.texte.lower().split():
                if word not in vocab:
                    vocab[word] = {"id": len(vocab), "nbr_occur": 0, "doc_freq": 0}
        return vocab

    def compute_tf(self, doc, vocab):
        """Calcule le TF (Term Frequency) pour un document."""
        tf = np.zeros(len(vocab))
        doc_words = doc.lower().split()
        for word in doc_words:
            if word in vocab:
                tf[vocab[word]["id"]] += 1
        return tf / len(doc_words) if len(doc_words) > 0 else tf

    def compute_idf(self, corpus, vocab):
        """Calcule l'IDF (Inverse Document Frequency) pour le corpus."""
        N = len(corpus)
        for doc in corpus.values():
            doc_words = set(doc.texte.lower().split())
            for word in doc_words:
                if word in vocab:
                    vocab[word]["doc_freq"] += 1

        for word, info in vocab.items():
            info["idf"] = math.log((N + 1) / (info["doc_freq"] + 1)) + 1

    def compute_tfidf(self, corpus, vocab):
        """Calcule la matrice TF-IDF pour le corpus."""
        N = len(corpus)
        M = len(vocab)
        tfidf_matrix = np.zeros((N, M))

        for i, doc in enumerate(corpus.values()):
            tf = self.compute_tf(doc.texte, vocab)
            for word, info in vocab.items():
                j = info["id"]
                tfidf_matrix[i, j] = tf[j] * info["idf"]

        return csr_matrix(tfidf_matrix)

    def cosine_similarity(self, vec1, vec2):
        """Calcule la similarité cosinus entre un vecteur dense et une matrice creuse."""
        vec1 = np.ravel(vec1)
        vec2 = np.ravel(vec2)
        dot_product = vec2.dot(vec1)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 != 0 else 0

    def search(self, query, top_k=5):
        """Recherche les documents les plus pertinents pour une requête donnée."""
        query = query.lower().split()
        query_vector = np.zeros((1, len(self.vocab)))
        for word in query:
            if word in self.vocab:
                word_id = self.vocab[word]["id"]
                query_vector[0, word_id] += 1

        similarities = np.zeros((self.mat_TF_IDF.shape[0], 1))
        for i in range(self.mat_TF_IDF.shape[0]):
            similarities[i] = self.cosine_similarity(
                query_vector, np.ravel((self.mat_TF_IDF.toarray())[i])
            )

        flat_array = similarities.flatten()
        clean_array = flat_array[~np.isnan(flat_array)]
        sorted_indices = np.argsort(clean_array)[::-1]

        results = []
        for idx in sorted_indices[:top_k]:
            results.append({"Document ID": idx, "Similarity": clean_array[idx]})

        return pd.DataFrame(results)
