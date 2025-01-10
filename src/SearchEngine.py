import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from Corpus import Corpus
from Document import Document
class SearchEngine:
    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab = self.build_vocab()

    def build_vocab(self):
        """Construit le vocabulaire à partir du corpus."""
        vocab = self.corpus.genererFreq().to_dict(orient="index")
        return vocab

    # Construire la matrice mat_TF
    def build_tf_matrix(self):
        num_words = len(self.vocab)
        data, rows, cols = [], [], []
        
        for i in range(1, self.corpus.ndoc + 1): 
            texte_nettoye = self.corpus.nettoyer_texte(self.corpus.id2doc[i].texte)
            mots = sorted(texte_nettoye.split())
        
            motsFreq = {mot: mots.count(mot) for mot in set(mots)}  # Compter occurrences de chaque mot dans le document
        
            for mot, occurence in motsFreq.items():
                if mot in self.vocab.keys():  
                    rows.append(i-1)  
                    cols.append(self.vocab[mot]["id"])  
                    data.append(occurence) 

        self.mat_TF = csr_matrix((data, (rows, cols)), shape=(self.corpus.ndoc, len(self.vocab)))
        # Mettre à jour les informations dans vocab
        for word, info in self.vocab.items():
            word_id = info["id"]
            # Nombre total d'occurrences du mot
            total_occurrences = self.mat_TF[:, word_id].sum()
            # Nombre de documents contenant le mot
            word_frequency = (self.mat_TF[:, word_id] > 0).sum()
            self.vocab[word]["nbr_occur"] = total_occurrences
            self.vocab[word]["nb_doc"] = word_frequency

        return self.mat_TF.toarray()

    def compute_tfidf(self):
        self.build_tf_matrix()
        """Calcule la matrice TF-IDF pour le corpus."""
        N = self.corpus.ndoc
        frequence_document = np.array((self.mat_TF > 0).sum(axis=0)).flatten()  # Nombre de documents contenant chaque mot
        IDF = np.log((N / (frequence_document+1)))
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
        self.compute_tfidf()
        """Recherche les documents les plus pertinents pour une requête donnée."""
        query = query.lower().split()
        query_vector = np.zeros((1, len(self.vocab)))
        for word in query:
            if word in self.vocab:
                word_id = self.vocab[word]["id"]
                query_vector[0, word_id] += 1
        print("QUERY",query_vector)
        similarities = np.zeros((self.mat_TF_IDF.shape[0], 1))
        for i in range(self.mat_TF_IDF.shape[0]):
            similarities[i] = self.cosine_similarity(
                query_vector, (self.mat_TF_IDF.toarray()[i])
            )

        flat_array = similarities.flatten()
        clean_array = flat_array[~np.isnan(flat_array)]
        sorted_indices = np.argsort(clean_array)[::-1]
        print(sorted_indices)
        results = []
        for id in sorted_indices[:top_k]:
            #results.append({"Document ID": idx, "Similarity": clean_array[idx]})
            results.append(self.corpus.id2doc[id+1])

        return results

if __name__ == "__main__":
    # Données de test : Liste de documents
    documents = [
        Document(titre="Doc1", auteur="Auteur1", texte="Ceci est un test de document.", date="2023-01-01"),
        Document(titre="Doc2", auteur="Auteur2", texte="Le test est une façon de vérifier le code.", date="2023-02-01"),
        Document(titre="Doc3", auteur="Auteur1", texte="Document avec des mots communs et uniques.", date="2023-03-01"),
        Document(titre="Doc4", auteur="Auteur3", texte="Ceci est un autre document pour les tests.", date="2023-04-01"),
    ]
    df = pd.read_csv("corpus/discours_US.csv",sep="\t")[:5]
    corpus = Corpus("US_discours")
    for i in range(df.shape[0]):
        row = df.loc[i,["speaker","text","date","link"]]
        text = row["text"]
        date = row["date"]
        auteur = row["speaker"]
        url = row["link"]
        phrases = text.split(".")
        for phrase in phrases:
            doc = Document("discours",auteur=auteur,date=date,url=url,texte=phrase)
            corpus.add(doc) 
    # Création du corpus
   

    # Ajout des documents dans le corpus
    """for doc in documents:
        corpus.add(doc)"""

    # Tester les fonctionnalités du corpus
    # Afficher les documents
    print("===== Documents =====")
    #corpus.show()

    search_engine = SearchEngine(corpus)
    #print(corpus.buildSearchString())
    #print(search_engine.build_vocab())
    
    #print(search_engine.build_tf_matrix())

    #print(search_engine.compute_tfidf().toarray())

    print(search_engine.search("students educators",2))

   # print(search_engine.cosine_similarity([0, 0, 0], [1, 2, 0]))"""




