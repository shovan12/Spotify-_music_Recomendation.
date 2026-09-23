# 🎵 Spotify Music Recommendation

A machine learning based Spotify-style music recommendation system built with **Python, Pandas, Scikit-learn, TF-IDF, Cosine Similarity, and Streamlit**.

## 🚀 Features

- 🔎 Search songs by song name or artist
- 🎧 Select a song from the dataset
- 🤖 Generate similar-song recommendations
- 📊 Display cosine similarity scores
- ⚡ Interactive Streamlit web interface
- 🧠 Lyric-based recommendation using TF-IDF

## 🛠️ Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit
- Pickle

## 📁 Project Structure

```text
Spotify-_music_Recomendation./
│
├── app.py
├── songs.pkl
├── similarity.pkl
├── tfidf.pkl
├── requirements.txt
├── Spotify_Music_Recommendation.ipynb
├── README.md
└── .gitignore
```

## ⚙️ How It Works

1. Song lyrics are processed and cleaned.
2. TF-IDF converts lyric text into numerical vectors.
3. Cosine similarity measures similarity between songs.
4. The most similar songs are selected.
5. Streamlit provides the interactive user interface.

## ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## 📌 Note

The recommendation model uses precomputed model files (`songs.pkl` and `similarity.pkl`). These files must be generated from the same dataset and kept aligned by row order.

## 👨‍💻 Author

**Shovan Pradhan**

GitHub: https://github.com/shovan12
