# Correction de G. Poux-Médard, 2021-2022
import re

from pandas import DataFrame

from model.Author import Author
from model.Document import Document


# =============== 2.7 : CLASSE CORPUS ===============
class Corpus:
    def __init__(self, nom):
        self.nom = nom
        self.authors = {}
        self.aut2id = {}
        self.id2doc = {}
        self.ndoc = 0
        self.naut = 0
        self._searchString = ""
        self.vocabulaire = set()
        self.freq = dict()

    def add(self, doc):
        if doc.auteur not in self.aut2id:
            self.naut += 1
            self.authors[self.naut] = Author(doc.auteur)
            self.aut2id[doc.auteur] = self.naut
        self.authors[self.aut2id[doc.auteur]].add(doc.texte)

        self.ndoc += 1
        self.id2doc[self.ndoc] = doc

    # =============== 2.8 : REPRESENTATION ===============
    def show(self, n_docs=-1, tri="abc"):
        docs = list(self.id2doc.values())
        if tri == "abc":  # Tri alphabétique
            docs = list(sorted(docs, key=lambda x: x.titre.lower()))[:n_docs]
        elif tri == "123":  # Tri temporel
            docs = list(sorted(docs, key=lambda x: x.date))[:n_docs]

        print("\n".join(list(map(repr, docs))))

    def __repr__(self):
        docs = list(self.id2doc.values())
        docs = list(sorted(docs, key=lambda x: x.titre.lower()))

        return "\n".join(list(map(str, docs)))

    # ===================TD6======================

    # Une fonction qui crée searchString
    def buildSearchString(self):
        if not self._searchString:
            textes = [self.id2doc[i].texte for i in range(1, self.ndoc + 1)]
            self._searchString = " ".join(textes)
        return self._searchString

    def search(self, keyword, context_size):
        # On construit la chaine si elle n'est pas encore construite
        self.buildSearchString()
        index = self._searchString.find(" " + keyword + " ")
        if index == -1:
            return "Mot-clé non trouvé."
        left_context = self._searchString[max(0, index - context_size) : index]
        word = self._searchString[index : index + len(keyword)]
        right_context = self._searchString[
            index + len(keyword) : min(
                index + len(keyword) + context_size, len(self._searchString)
            )
        ]
        passage = left_context + word + right_context
        return passage

    # La fonction concorde
    def concorde(self, keyword, context_size):
        tab = {"contexte gauche": [], "motif trouve": [], "contexte droit": []}
        self.buildSearchString()
        occurrences = re.finditer(re.escape(keyword), self._searchString)
        indices = [match.start() for match in occurrences]
        for i in indices:
            tab["contexte gauche"].append(
                self._searchString[max(0, i - context_size) : i]
            )
            tab["motif trouve"].append(self._searchString[i : i + len(keyword)])
            tab["contexte droit"].append(
                self._searchString[
                    i + len(keyword) : min(
                        i + len(keyword) + context_size, len(self._searchString)
                    )
                ]
            )
        return DataFrame.from_dict(tab)

    def stats(self, n=10):
        self.generer_vocabulaire()
        freq_df = self.genererFreq()
        nb_mots_differents = len(self.vocabulaire)
        print(f"Nombre de mots différents dans le corpus : {nb_mots_differents}")
        mots_plus_frequents = freq_df.sort_values(
            by="Term Frequency", ascending=False
        ).head(n)
        print(f"\nLes {n} mots les plus fréquents :")
        print(mots_plus_frequents)

    # Une fonction qui permet de nettoyer un texte
    def nettoyer_texte(self, texte):
        texte = texte.lower()
        texte = re.sub(r"\n", " ", texte)
        texte = re.sub(r"[^\w\s]", "", texte)
        texte = re.sub(r"\d+", "", texte)
        return texte

    # Une fonction qui génère le vocabulaire de notre corpus
    def generer_vocabulaire(self):
        texte = self.buildSearchString()
        mots_liste = []
        texte_nettoye = self.nettoyer_texte(texte)
        mots = re.split(r"\s+|[,.!?;:\'\"(){}\[\]]+", texte_nettoye)
        mots_liste.extend(mots)
        self.vocabulaire = sorted(set(mots_liste))
        return self.vocabulaire

    # Une fonction qui prend en paramètre un mot et retourne le nombre de doc dans lequel il apparait
    def nbdoc2mot(self, mot):
        count = 0
        for i in range(1, self.ndoc + 1):
            if mot in self.id2doc[i].texte.split(" "):
                count += 1
        return count

    def genererFreq(self):
        self.buildSearchString()
        self.generer_vocabulaire()
        id = 0
        for mot in self.vocabulaire:
            term_frequency = self._searchString.lower().split(" ").count(mot)
            document_frequency = self.nbdoc2mot(mot)
            self.freq[mot] = [id, term_frequency, document_frequency]
            id += 1
        df = DataFrame.from_dict(
            self.freq,
            orient="index",
            columns=["id", "Term Frequency", "Document Frequency"],
        )
        return df


class SingletonCorpus(Corpus):
    _instance = None

    @staticmethod
    def get_instance(nom="DefaultCorpus"):
        if SingletonCorpus._instance is None:
            SingletonCorpus._instance = SingletonCorpus(nom)
        return SingletonCorpus._instance

    def __init__(self, nom):
        if SingletonCorpus._instance is not None:
            raise Exception("This class is a singleton!")
        super().__init__(nom)

    def search(self, keyword):
        return super().search(keyword)


if __name__ == "__main__":
    # Données de test : Liste de documents
    documents = [
        Document(
            titre="Doc1",
            auteur="Auteur1",
            texte="Ceci est un test de document.",
            date="2023-01-01",
        ),
        Document(
            titre="Doc2",
            auteur="Auteur2",
            texte="Le test est une façon de vérifier le code.",
            date="2023-02-01",
        ),
        Document(
            titre="Doc3",
            auteur="Auteur1",
            texte="Document avec des mots communs et uniques.",
            date="2023-03-01",
        ),
        Document(
            titre="Doc4",
            auteur="Auteur3",
            texte="Ceci est un autre document pour les tests.",
            date="2023-04-01",
        ),
    ]

    # Création du corpus
    corpus = Corpus(nom="TestCorpus")

    # Ajout des documents dans le corpus
    for doc in documents:
        corpus.add(doc)

    # Tester les fonctionnalités du corpus
    # Afficher les documents
    print("===== Documents =====")
    corpus.show()

    # Rechercher un mot-clé
    print("\n===== Recherche =====")
    print(corpus.search("test", context_size=10))

    # Générer le vocabulaire
    print("\n===== Vocabulaire =====")
    corpus.generer_vocabulaire()
    print(corpus.vocabulaire)

    # Générer les fréquences
    print("\n===== Fréquences =====")
    freq_df = corpus.genererFreq()
    print(freq_df)

    # Concorde pour un mot-clé
    print("\n===== Concorde =====")
    concorde_df = corpus.concorde("test", context_size=10)
    print(concorde_df)

    # Statistiques sur les mots les plus fréquents
    print("\n===== Statistiques =====")
    corpus.stats(n=5)
