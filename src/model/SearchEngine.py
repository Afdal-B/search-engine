import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from model.Corpus import Corpus
from model.Document import Document

class SearchEngine:
    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab = self.build_vocab()  # Construire le vocabulaire

    def build_vocab(self):
        """Construit le vocabulaire à partir du corpus."""
        vocab = self.corpus.genererFreq().to_dict(orient="index")
        return vocab

    def build_tf_matrix(self):
        """Construire la matrice TF (Term Frequency)."""
        num_words = len(self.vocab)
        data, rows, cols = [], [], []

        for i in range(1, self.corpus.ndoc + 1):
            texte_nettoye = self.corpus.nettoyer_texte(self.corpus.id2doc[i].texte)
            mots = sorted(texte_nettoye.split())

            motsFreq = {mot: mots.count(mot) for mot in set(mots)}  # Compter occurrences de chaque mot

            for mot, occurence in motsFreq.items():
                if mot in self.vocab.keys():
                    rows.append(i - 1)
                    cols.append(self.vocab[mot]["id"])
                    data.append(occurence)

        self.mat_TF = csr_matrix((data, (rows, cols)), shape=(self.corpus.ndoc, len(self.vocab)))

        # Mettre à jour les informations dans vocab
        for word, info in self.vocab.items():
            word_id = info["id"]
            total_occurrences = self.mat_TF[:, word_id].sum()  # Nombre total d'occurrences du mot
            word_frequency = (self.mat_TF[:, word_id] > 0).sum()  # Nombre de documents contenant le mot
            self.vocab[word]["nbr_occur"] = total_occurrences
            self.vocab[word]["nb_doc"] = word_frequency

        return self.mat_TF.toarray()

    def compute_tfidf(self):
        """Calcule la matrice TF-IDF pour le corpus."""
        self.build_tf_matrix()
        N = self.corpus.ndoc
        frequence_document = np.array((self.mat_TF > 0).sum(axis=0)).flatten()  # Nombre de documents contenant chaque mot
        IDF = np.log((N / (frequence_document + 1)))
        self.mat_TF_IDF = self.mat_TF.multiply(IDF)
        return self.mat_TF_IDF

    def cosine_similarity(self, vec1, vec2):
        """Calcule la similarité cosinus entre un vecteur dense et une matrice creuse."""
        vec1 = np.ravel(vec1)
        vec2 = np.ravel(vec2)
        dot_product = vec2.dot(vec1)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if np.isnan(dot_product) or np.isnan(norm1) or np.isnan(norm2):
            return 0
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)

    def search(self, query, top_k=5):
        """Recherche les documents les plus pertinents pour une requête donnée."""
        self.compute_tfidf()
        query = query.lower().split()
        query_vector = np.zeros((1, len(self.vocab)))
        for word in query:
            if word in self.vocab:
                word_id = self.vocab[word]["id"]
                query_vector[0, word_id] += 1

        similarities = np.zeros((self.mat_TF_IDF.shape[0], 1))
        for i in range(self.mat_TF_IDF.shape[0]):
            similarities[i] = self.cosine_similarity(query_vector, (self.mat_TF_IDF.toarray()[i]))

        flat_array = similarities.flatten()
        clean_array = flat_array[~np.isnan(flat_array)]
        sorted_indices = np.argsort(clean_array)[::-1]

        results = []
        for id in sorted_indices[:top_k]:
            results.append(self.corpus.id2doc[id + 1])

        return pd.DataFrame(results)

if __name__ == "__main__":
    pass
