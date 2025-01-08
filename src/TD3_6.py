# Correction de G. Poux-Médard, 2021-2022
# =============== PARTIE 1 =============
# =============== 1.1 : REDDIT ===============
# Library
import datetime
import pickle
import urllib
import urllib.request

import praw
import xmltodict

from model.Author import Author
from model.Corpus import Corpus
from model.Document import ArxivDocument, RedditDocument


# Fonction affichage hiérarchie dict
def showDictStruct(d):
    def recursivePrint(d, i):
        for k in d:
            if isinstance(d[k], dict):
                print("-" * i, k)
                recursivePrint(d[k], i + 2)
            else:
                print("-" * i, k, ":", d[k])

    recursivePrint(d, 1)


def fetch_reddit_posts(limit=100):
    # Identification
    reddit = praw.Reddit(
        client_id="J84kxymaZdKMDdplOHKoEQ",
        client_secret="TcR1l8FaeqUgEcWYmqu30pXYP8MRog",
        user_agent="redit_scrapping",
    )

    # Requête
    hot_posts = reddit.subreddit("all").hot(limit=limit)

    # Récupération du texte
    docs = []
    docs_bruts = []
    afficher_cles = False
    for i, post in enumerate(hot_posts):
        if i % 10 == 0:
            print("Reddit:", i, "/", limit)
        if afficher_cles:  # Pour connaître les différentes variables et leur contenu
            for k, v in post.__dict__.items():
                pass
                print(k, ":", v)

        if post.selftext != "":  # Osef des posts sans texte
            pass
        docs.append(post.selftext.replace("\n", " "))
        docs_bruts.append(("Reddit", post))

    return docs, docs_bruts


def fetch_arxiv_docs(query_terms, max_results=50):
    # Requête
    url = f'http://export.arxiv.org/api/query?search_query=all:{"+".join(query_terms)}&start=0&max_results={max_results}'
    data = urllib.request.urlopen(url)

    # Format dict (OrderedDict)
    data = xmltodict.parse(data.read().decode("utf-8"))

    # Ajout résumés à la liste
    docs = []
    docs_bruts = []
    for i, entry in enumerate(data["feed"]["entry"]):
        if i % 10 == 0:
            print("ArXiv:", i, "/", max_results)
        docs.append(entry["summary"].replace("\n", ""))
        docs_bruts.append(("ArXiv", entry))

    return docs, docs_bruts


def process_documents(docs, docs_bruts):
    print(f"# docs avec doublons : {len(docs)}")
    docs = list(set(docs))
    print(f"# docs sans doublons : {len(docs)}")

    for i, doc in enumerate(docs):
        print(
            f"Document {i}\t# caractères : {len(doc)}\t# mots : {len(doc.split(' '))}\t# phrases : {len(doc.split('.'))}"
        )
        if len(doc) < 100:
            docs.remove(doc)

    return docs, docs_bruts


def create_collection(docs_bruts):
    collection = []
    for nature, doc in docs_bruts:
        if nature == "ArXiv":
            titre = doc["title"].replace("\n", "")
            try:
                authors = ", ".join([a["name"] for a in doc["author"]])
            except Exception:
                authors = doc["author"]["name"]
            summary = doc["summary"].replace("\n", "")
            date = datetime.datetime.strptime(
                doc["published"], "%Y-%m-%dT%H:%M:%SZ"
            ).strftime("%Y/%m/%d")

            doc_classe = ArxivDocument(titre, authors, date, doc["id"], summary)
            collection.append(doc_classe)

        elif nature == "Reddit":
            titre = doc.title.replace("\n", "")
            auteur = str(doc.author)
            date = datetime.datetime.fromtimestamp(doc.created).strftime("%Y/%m/%d")
            url = "https://www.reddit.com/" + doc.permalink
            texte = doc.selftext.replace("\n", "")

            doc_classe = RedditDocument(titre, auteur, date, url, texte)
            collection.append(doc_classe)

    return collection


def create_author_dict(collection):
    authors = {}
    aut2id = {}
    num_auteurs_vus = 0

    for doc in collection:
        if doc.auteur not in aut2id:
            num_auteurs_vus += 1
            authors[num_auteurs_vus] = Author(doc.auteur)
            aut2id[doc.auteur] = num_auteurs_vus

        authors[aut2id[doc.auteur]].add(doc.texte)

    return authors, aut2id


def save_corpus(corpus, filename="corpus/corpus.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(corpus, f)


def load_corpus(filename="corpus/corpus.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


def main():
    # Récupérer les posts Reddit
    reddit_docs, reddit_docs_bruts = fetch_reddit_posts()

    # Récupérer les documents ArXiv
    query_terms = ["clustering", "Dirichlet"]
    arxiv_docs, arxiv_docs_bruts = fetch_arxiv_docs(query_terms)

    # Combiner les documents
    docs = reddit_docs + arxiv_docs
    docs_bruts = reddit_docs_bruts + arxiv_docs_bruts

    # Traiter les documents
    docs, docs_bruts = process_documents(docs, docs_bruts)

    # Créer la collection
    collection = create_collection(docs_bruts)

    # Créer le dictionnaire des auteurs
    authors, aut2id = create_author_dict(collection)

    # Créer le corpus
    corpus = Corpus("Mon corpus")
    for doc in collection:
        corpus.add(doc)

    # Sauvegarder le corpus
    save_corpus(corpus)

    # Charger le corpus
    corpus_f = load_corpus()

    print("=========CORPUS-File==========")
    print(corpus_f)

    print("=========CORPUS-Brut==========")
    print(corpus)

    print("UTILISATION DE LA METHODE CONCORDE\n")
    print(corpus_f.concorde("the", 5))
    print("UTILISATION DE LA METHODE STATS\n")
    print(corpus_f.stats(n=5))
    print("UTILISATION DE LA METHODE GENERERFREQ\n")
    print(corpus_f.genererFreq())


if __name__ == "__main__":
    main()
