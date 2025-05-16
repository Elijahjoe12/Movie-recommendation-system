import pandas as pd
import re
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")



# Load and sample dataset
try:
    movies = pd.read_csv("movies.csv")
    logging.info("✅ Dataset loaded and sampled: %d rows", len(movies))
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

#feature selection
features_sel = ['genres', 'keywords', 'tagline', 'cast', 'director']


# replacing null values with null string
for feature in features_sel:
    movies[feature] = movies[feature].fillna('')


# combining the selected features
comb_features = movies['genres']+' '+movies['keywords']+' '+movies['tagline']+' '+movies['cast']+' '+movies['director']



# Vectorization
logging.info("🔠 Vectorizing using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
feature_vectors = tfidf.fit_transform(comb_features)
logging.info("✅ feature vectors matrix shape: %s", feature_vectors.shape)

# Cosine similarity
logging.info("📐 Calculating cosine similarity...")
cos_sim = cosine_similarity(feature_vectors)
logging.info("✅ Cosine similarity matrix generated.", cos_sim.shape)

# Save everything
joblib.dump(movies, 'movies.pkl')
joblib.dump(feature_vectors, 'feature_vectors.pkl')
joblib.dump(cos_sim, 'cosine_sim.pkl')
logging.info("💾 Data saved to disk.")

logging.info("✅ Preprocessing complete.")