# AI-Works

A collection of practical AI and Machine Learning implementations covering natural language processing, regression, classification, and recommendation systems. Each notebook contains hands-on code with real-world examples.

## Overview

This repository includes various machine learning and AI implementations created for learning and experimentation. Each notebook is self-contained with working examples and demonstrations of different algorithms and techniques.

## Repository Structure

```
AI-Works/
├── README.md
└── Codes/
    ├── Basics_of_Natural_Language_Processing_(NLP).ipynb
    ├── Linear_Regression.ipynb
    ├── Classification_using_K-Nearest_Neighbors_Random_Forest_and_MLP_(KNN).ipynb
    ├── Ngram_based_next_word_predicetion.ipynb
    ├── TextRank_for_keyword_Extraction.ipynb
    ├── Reinforcement_Learning_(RL).ipynb
    ├── Collaborative_Filtering_based_recommendation_system.ipynb
    └── Recommendation_System_using_content_based.ipynb
```

## Contents

### 1. Basics of Natural Language Processing (NLP)

**File:** `Basics_of_Natural_Language_Processing_(NLP).ipynb`

Fundamental concepts of Natural Language Processing using spaCy library:
- Text tokenization and word splitting
- String and punctuation handling
- Regex operations for text processing
- Named Entity Recognition (NER) - extracting entities from text
- Part-of-Speech (POS) tagging
- Stopwords identification and filtering
- Stemming using Porter Stemmer
- Lemmatization for word normalization
- Understanding token objects vs string objects

**Key Libraries:** spaCy, NLTK, regex, string

---

### 2. Linear Regression

**File:** `Linear_Regression.ipynb`

Simple linear regression implementation from scratch:
- Function to estimate regression coefficients (b0 and b1)
- Calculation of means, cross-deviation, and deviation
- Plotting regression line with scatter plot visualization
- Using sample data for prediction
- Understanding line fitting principles

**Key Libraries:** NumPy, Matplotlib

**Approach:** Mathematical computation of regression coefficients using formula-based approach

---

### 3. Classification using K-Nearest Neighbors (KNN), Random Forest, and Multi-Layer Perceptron (MLP)

**File:** `Classification_using_K-Nearest_Neighbors_Random_Forest_and_MLP_(KNN).ipynb`

Classification experiments comparing three algorithms on two datasets (Digits and Wine):

**Algorithms Compared:**
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Multi-Layer Perceptron (MLP) Neural Network

**Features:**
- Pipeline creation for preprocessing and model integration
- GridSearchCV for hyperparameter tuning
- 5-fold cross-validation
- Train-test split (80-20)
- Performance metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix visualization using Seaborn
- Execution time comparison (training and inference)
- Bar chart visualization for model performance
- Testing on both Digits dataset (handwritten digits) and Wine dataset

**Key Libraries:** scikit-learn, NumPy, Pandas, Matplotlib, Seaborn

**Workflow:**
1. Load and visualize dataset
2. Split data into train and test sets
3. Create pipelines with scaling and classifiers
4. Hyperparameter tuning using GridSearchCV
5. Evaluate all models with metrics
6. Compare results across models
7. Generate confusion matrices and performance plots

---

### 4. N-gram Based Next Word Prediction

**File:** `Ngram_based_next_word_predicetion.ipynb`

Simple language model for predicting next words using N-grams:
- Text tokenization using NLTK
- Bigram generation (2-word sequences)
- Word frequency counting with Counter
- Dictionary mapping using defaultdict
- Prediction function returning most frequent next word
- Building model from corpus

**Key Libraries:** NLTK, collections

**Approach:** Statistical model that predicts the next word based on current word by finding the most frequently occurring successor in training data

---

### 5. TextRank for Keyword Extraction

**File:** `TextRank_for_keyword_Extraction.ipynb`

Keyword extraction using TextRank algorithm with graph-based approach:
- Text tokenization and lowercasing
- Stopword removal from text
- Punctuation filtering
- Graph creation with words as nodes and edges between co-occurring words
- PageRank algorithm application for scoring
- Extraction of top N keywords

**Key Libraries:** NLTK, NetworkX

**Algorithm:**
1. Preprocess text (tokenize, remove stopwords and punctuation)
2. Build co-occurrence graph within sliding window
3. Apply PageRank to compute word importance scores
4. Return top-ranked keywords

---

### 6. Reinforcement Learning (RL)

**File:** `Reinforcement_Learning_(RL).ipynb`

Q-Learning algorithm implementation for finding optimal path in a graph:
- Graph creation and visualization using NetworkX
- Reward matrix definition for state-action pairs
- Q-matrix initialization
- Q-learning algorithm with gamma discount factor (0.75)
- Functions to determine available actions from a state
- Random action selection
- Iterative Q-matrix updates
- Learning process over multiple episodes
- Goal-directed navigation

**Key Libraries:** NumPy, NetworkX, Pylab

**Key Concepts:**
- State-action reward definition
- Q-learning update formula: Q(s,a) = R(s,a) + gamma * max(Q(a,:))
- Reward matrix setup
- Discount factor (gamma)
- Iterative learning towards goal state

---

### 7. Collaborative Filtering Based Recommendation System

**File:** `Collaborative_Filtering_based_recommendation_system.ipynb`

User-based collaborative filtering approach for recommendations:
- Loading rating data from CSV
- Creating user-item rating matrix using pivot table
- Handling missing values (filling with 0)
- Computing cosine similarity between users
- Building user similarity dataframe
- Matrix visualization

**Key Libraries:** Pandas, scikit-learn (cosine_similarity)

**Approach:**
1. Create user-item rating matrix from raw data
2. Calculate similarity scores between all user pairs using cosine similarity
3. Use similar users' ratings to recommend items to target user
4. Based on the principle: if two users have similar rating patterns, they likely have similar preferences

**Dataset:** Requires ratings.csv file with userId, movieId, and rating columns

---

### 8. Content-Based Recommendation System

**File:** `Recommendation_System_using_content_based.ipynb`

Movie recommendation system based on content similarity:
- Loading movie data with genres and overview
- Handling missing values in features
- Combining genre and overview as content features
- TF-IDF vectorization of content
- Computing cosine similarity matrix between all movies
- Movie indexing by title with duplicate handling
- Recommendation function returning top-N similar movies

**Key Libraries:** Pandas, scikit-learn (TfidfVectorizer, cosine_similarity)

**Workflow:**
1. Load movie dataset with title, genres, and overview
2. Preprocess: fill missing values, combine features
3. Apply TF-IDF vectorizer to convert text to numerical vectors
4. Calculate cosine similarity between all movies
5. Create title-to-index mapping (dropping duplicates)
6. For a given movie, find similar movies using similarity scores
7. Return top N recommendations

**Approach:** Content-based filtering where movies are recommended based on how similar they are in terms of genre and plot description

**Dataset:** Requires movies.csv file with columns: original_title, genres, overview

---

## Installation and Setup

### Requirements

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab for running notebooks
- Ability to run on Google Colab (recommended for beginners)

### Dependencies

All required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk spacy networkx
```

### Additional Setup

For spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

For NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

## Usage

1. Clone this repository
2. Navigate to the Codes folder
3. Open any notebook in Jupyter, JupyterLab, or Google Colab
4. Run cells from top to bottom
5. Each notebook is independent and can be executed without running others

All notebooks have "Open in Colab" links at the top for cloud-based execution without local setup.

## Dataset Requirements

Some notebooks require CSV data files:
- `Collaborative_Filtering_based_recommendation_system.ipynb` - requires ratings.csv
- `Recommendation_System_using_content_based.ipynb` - requires movies.csv

These files should be placed in the same directory or loaded from appropriate paths.

## Key Concepts Covered

- Natural Language Processing fundamentals
- Text preprocessing, tokenization, and normalization
- Named Entity Recognition and Part-of-Speech tagging
- Stemming and lemmatization
- Statistical regression analysis
- Supervised learning and classification
- K-Nearest Neighbors algorithm
- Decision trees and ensemble methods (Random Forest)
- Neural networks (Multi-Layer Perceptron)
- Hyperparameter tuning and cross-validation
- N-gram language models
- Graph-based algorithms (PageRank, TextRank)
- Reinforcement learning and Q-Learning
- Similarity metrics (cosine similarity)
- Recommendation systems (collaborative and content-based filtering)
- Model evaluation metrics (accuracy, precision, recall, F1-score)

## Learning Path

**Beginner Level:**
- Start with Basics of NLP
- Move to Linear Regression (simple mathematical ML)
- Explore N-gram Word Prediction

**Intermediate Level:**
- Learn Classification algorithms (KNN, Random Forest, MLP)
- Understand TextRank for keyword extraction
- Study content-based recommendations

**Advanced Level:**
- Implement Reinforcement Learning with Q-Learning
- Develop collaborative filtering systems
- Experiment with different hyperparameters

## Technologies and Libraries Used

- **Machine Learning:** scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Natural Language Processing:** NLTK, spaCy
- **Graph Processing:** NetworkX
- **Numeric Computing:** NumPy (matrices, arrays)
- **IDE:** Jupyter Notebook, Google Colab

## Important Notes

- Notebooks focus on learning and understanding algorithms
- Code includes explanations and comments
- Sample datasets are used for demonstration
- Visualizations help understand model behavior
- Hyperparameter values in GridSearchCV are examples and can be modified
- Some notebooks require external data files (CSV) which should be provided
- All code runs on Google Colab without additional configuration

## Author

Milan Jani

## License

This repository is open for educational purposes.
