import joblib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🔁 Loading data...")
try:
    movies = joblib.load('movies.pkl')
    cos_sim = joblib.load('cosine_sim.pkl')
    logging.info("✅ Data loaded successfully.")
except Exception as e:
    logging.error("❌ Failed to load required files: %s", str(e))
    raise e


def recommend_movies(movie_name, top_n=10):
    logging.info("🎬 Recommending songs for: '%s'", movie_name)
    index = movies[movies['title'].str.lower() == movie_name.lower()].index
    if len(index) == 0:
        logging.warning("⚠️ movie not found in dataset.")
        return None
    index = index[0]
    sim_scores = list(enumerate(cos_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    logging.info("✅ Top %d recommendations ready.", top_n)
    # Create DataFrame with clean serial numbers starting from 1
    result_movies = movies[['title', 'genres']].iloc[movie_indices].reset_index(drop=True)
    result_movies.index = result_movies.index + 1  # Start from 1 instead of 0
    result_movies.index.name = "S.No."

    return result_movies