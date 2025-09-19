# Movie Recommender — Content-Based (TMDB 5000)

A production-ready **content-based movie recommendation system** that analyzes **5,000+ films** from the TMDB 5000 dataset, fusing **genres, keywords, top cast, director, and overview** into a single feature space and retrieving **top-5 similar movies** via **cosine similarity** on vectorized tags.

> **Highlights:** preprocessed & serialized artifacts (`movies.pkl`, `similarity.pkl`) for **sub-200ms lookups**; clean notebook + ready-to-use functions for quick integration into an API/UI.

---

## Features
- **Content-based** recommendations (no user ratings required) using combined metadata (**genres, keywords, cast[3], director, overview**).
- **NLP vectorization** with `CountVectorizer` (stop-word removal, max features) and **cosine similarity**.
- **Precomputation & caching:** saves `movies.pkl` and `similarity.pkl` for instant inference.
- **Deterministic ranking:** stable top-K (excludes the queried movie).
- **Extensible:** drop-in ready for Flask/FastAPI/Streamlit UI.

---

## How it Works
1. **Load & Merge:** TMDB 5000 Movies and Credits CSVs are read and merged.  
2. **Parse & Clean:** JSON-like strings in `genres`, `keywords`, `cast`, `crew` are parsed; **top 3 cast** and **director** are kept.  
3. **Normalize Text:** lowercase, strip spaces in multi-word tokens (e.g., `SamWorthington`), tokenize `overview`.  
4. **Build `tags`:** concatenate cleaned tokens from all fields into a single text feature.  
5. **Vectorize:** transform `tags` using `CountVectorizer` (`max_features=5000`, `stop_words='english'`).  
6. **Similarity:** compute **cosine similarity** between all movie vectors (N×N matrix).  
7. **Recommend:** for a given title, retrieve and sort most similar movies and return **top-5**.  

---

## Setup & Quickstart
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install numpy pandas scikit-learn
```
---



## Data
Place the two CSV files in a `data/` folder:

- `tmdb_5000_movies.csv`  
- `tmdb_5000_credits.csv`  

Run the notebook once to generate `movies.pkl` and `similarity.pkl`.

---

## Benchmarks
- Dataset size: **~5,000 movies**  
- Vector dimension: **≤ 5,000 features**  
- Lookup time: **~100–200ms** for top-5  

---

## Extensions
- Replace `CountVectorizer` with **TF-IDF** or embeddings (`sentence-transformers`).  
- Hybrid approach: combine with **collaborative filtering**.  
- Add **fuzzy title search** for better UX.  
- Build a **Streamlit/React UI**.  

---

## Acknowledgments
- **TMDB 5000 dataset**.  
- **scikit-learn** for vectorization & similarity utilities.  
