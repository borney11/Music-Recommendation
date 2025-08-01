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

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

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
    if os.path.exists(DF_FILE) and os.path.exists(TFIDF_FILE) and os.path.exists(COSINE_FILE):
        logging.info("✅ Loading preprocessed data from PKL files...")
        df = joblib.load(DF_FILE)
        tfidf_matrix = joblib.load(TFIDF_FILE)
        cosine_sim = joblib.load(COSINE_FILE)
    else:
        raise FileNotFoundError("Pickle files missing.")
except Exception as e:
    logging.warning("⚠️ PKL load failed: %s", e)

    if os.path.exists(CSV_FILE):
        logging.info("➡️ Loading df_cleaned.csv instead of PKL...")
        df = pd.read_csv(CSV_FILE)

        if 'cleaned_text' not in df.columns:
            if 'text' in df.columns:
                logging.warning("⚠️ 'cleaned_text' missing, regenerating from 'text'...")
                df['cleaned_text'] = df['text'].apply(preprocess_text)
            else:
                logging.error("❌ Neither 'cleaned_text' nor 'text' found in CSV. Using raw dataset.")
                df = pd.read_csv(DATA_FILE).sample(10000)
                df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)
                df['cleaned_text'] = df['text'].apply(preprocess_text)
    else:
        logging.info("🚀 No PKL/CSV found, preprocessing from raw dataset...")
        df = pd.read_csv(DATA_FILE).sample(10000)
        df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)
        df['cleaned_text'] = df['text'].apply(preprocess_text)

# ✅ TF-IDF and cosine similarity
logging.info("🔠 Vectorizing using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

logging.info("📐 Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("✅ Cosine similarity matrix generated.")

# ✅ Save all
joblib.dump(df, DF_FILE)
joblib.dump(tfidf_matrix, TFIDF_FILE)
joblib.dump(cosine_sim, COSINE_FILE)
df.to_csv(CSV_FILE, index=False)

logging.info("🎉 Preprocessing complete.")
