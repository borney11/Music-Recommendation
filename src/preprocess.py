import os
import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

DATA_FILE = "spotify_millsongdata.csv"
DF_FILE = "df_cleaned.pkl"
TFIDF_FILE = "tfidf_matrix.pkl"
COSINE_FILE = "cosine_sim.pkl"
CSV_FILE = "df_cleaned.csv"

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

try:
    # ✅ Attempt to load pickle files
    if all(os.path.exists(f) for f in [DF_FILE, TFIDF_FILE, COSINE_FILE]):
        logging.info("✅ Loading preprocessed data from PKL files...")
        df = joblib.load(DF_FILE)
        tfidf_matrix = joblib.load(TFIDF_FILE)
        cosine_sim = joblib.load(COSINE_FILE)
    else:
        raise FileNotFoundError("Pickle files missing.")
except Exception as e:
    logging.warning(f"⚠️ PKL load failed: {e}")

    # ✅ If CSV exists, use it
    if os.path.exists(CSV_FILE):
        logging.info("➡️ Loading df_cleaned.csv instead of PKL...")
        df = pd.read_csv(CSV_FILE)
        logging.info("🔠 Regenerating TF-IDF and cosine similarity from CSV...")
        tfidf = TfidfVectorizer(max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    else:
        logging.info("🚀 Preprocessing data from scratch...")
        df = pd.read_csv(DATA_FILE).sample(10000)
        logging.info("✅ Dataset loaded: %d rows", len(df))

        df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

        logging.info("🧹 Cleaning text...")
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        logging.info("✅ Text cleaned.")

        logging.info("🔠 Vectorizing using TF-IDF...")
        tfidf = TfidfVectorizer(max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
        logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

        logging.info("📐 Calculating cosine similarity...")
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        logging.info("✅ Cosine similarity matrix generated.")

        # ✅ Save both PKL and CSV
        joblib.dump(df, DF_FILE)
        joblib.dump(tfidf_matrix, TFIDF_FILE)
        joblib.dump(cosine_sim, COSINE_FILE)
        df.to_csv(CSV_FILE, index=False)
        logging.info("💾 Preprocessed data saved to PKL and CSV.")

logging.info("🎉 Preprocessing complete.")
