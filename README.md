# SEARCH-ENGINE

## Description

Ce projet implémente un moteur de recherche qui manipule et analyse un corpus de documents. Le projet est organisé de manière modulaire, avec des classes pour représenter les documents, les auteurs, le corpus, et le moteur de recherche.

## Fonctionnalités principales

- Gestion de corpus : Importation, exportation et traitement des documents.
- Moteur de recherche : Recherche avancée dans le corpus basé sur différents critères.
- Analyse de documents : Gestion des métadonnées et traitement des contenus.

## Arborescence du projet

```bash
SEARCH-ENGINE/
├── corpus/
│ ├── corpus.pkl # Fichier contenant le corpus sérialisé
│ ├── discours_US.csv # Exemple de corpus sous format CSV
│ ├── out.pkl # Fichier de sortie (résultats ou corpus traité)
├── src/
│ ├── model/ # Contient les définitions des classes principales
│ │ ├── __init__.py # Initialisation du module
│ │ ├── Author.py # Classe représentant les auteurs
│ │ ├── Corpus.py # Classe représentant un corpus de documents
│ │ ├── Document.py # Classe représentant un document
│ │ ├── SearchEngine.py # Classe implémentant le moteur de recherche
│ ├── TD3_6.py # Script pour les l'exécution des TD 3 à 6
│ ├── TD7.py # Script pour les l'exécution du TD 7
│ ├── td8.ipynb # Notebook principal pour la démonstration
├── .gitignore # Liste des fichiers/dossiers ignorés par Git
├── LICENSE # Licence du projet
├── README.md # Documentation du projet
├── requirements.txt # Liste des dépendances Python
```

## Prérequis

### Environnement Python

Ce projet a été développé avec Python 3.9.11 et plus. Assurez-vous d'avoir installé un environnement Python compatible avant de commencer.

### Dépendances

Les dépendances nécessaires sont listées dans le fichier requirements.txt. Pour les installer, exécutez :

```bash
pip install -r requirements.txt
```

## Utilisation

### Lancement du projet

1. Chargement du corpus : Le fichier corpus.pkl contient un corpus de documents préalablement sérialisé. Vous pouvez également charger le fichier discours_US.csv pour un corpus brut.
2. Moteur de recherche : Implémenté dans SearchEngine.py, ce module permet d'effectuer des recherches dans le corpus.
3. Analyse des documents : Utilisez les classes Corpus et Document pour manipuler et analyser les documents.

### Exemple d'utilisation

Un exemple d'utilisation est fourni dans le notebook td8.ipynb, où vous pouvez exécuter et tester les fonctionnalités principales.
