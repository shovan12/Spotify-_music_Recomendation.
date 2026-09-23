import streamlit as st
import pickle
import pandas as pd
import re

st.set_page_config(
    page_title="Spotify Music Recommendation",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_data():
    with open("similarity.pkl", "rb") as f:
        similarity = pickle.load(f)
    with open("songs.pkl", "rb") as f:
        songs = pickle.load(f)
    return similarity, songs

try:
    similarity, df = load_data()
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading project files: {e}")
    st.stop()

if not isinstance(df, pd.DataFrame):
    st.error("❌ songs.pkl does not contain a Pandas DataFrame.")
    st.stop()

required_columns = ["song", "artist"]
missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    st.error(f"❌ Missing columns: {missing_columns}")
    st.stop()

df = df.reset_index(drop=True)
df["song"] = df["song"].fillna("").astype(str).str.strip()
df["artist"] = df["artist"].fillna("").astype(str).str.strip()

def normalize_text(text):
    text = str(text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("&", "and")
    text = text.replace("+", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["song_search"] = df["song"].apply(normalize_text)
df["artist_search"] = df["artist"].apply(normalize_text)

try:
    similarity_length = len(similarity)
except Exception:
    st.error("❌ similarity.pkl is not a valid similarity matrix.")
    st.stop()

if similarity_length != len(df):
    st.error(
        f"❌ Data mismatch detected. Songs: {len(df)} | "
        f"Similarity matrix: {similarity_length}"
    )
    st.stop()

st.title("🎵 Spotify Music Recommendation")
st.write(
    "Discover songs similar to your favorite track "
    "using TF-IDF and Cosine Similarity."
)
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎵 Songs", f"{len(df):,}")
with col2:
    st.metric("🎤 Artists", f"{df['artist'].nunique():,}")
with col3:
    st.metric("🤖 Method", "TF-IDF + Cosine")

st.divider()
st.subheader("🔎 Find Your Song")

search_text = st.text_input(
    "Search by song name or artist",
    placeholder="Example: The End Of Camelot / Peter Cetera"
)

if search_text.strip():
    search_query = normalize_text(search_text)
    search_words = search_query.split()

    phrase_match = (
        df["song_search"].str.contains(search_query, regex=False, na=False)
        | df["artist_search"].str.contains(search_query, regex=False, na=False)
    )
    filtered_df = df[phrase_match]

    if len(filtered_df) == 0:
        word_match = pd.Series(True, index=df.index)
        for word in search_words:
            word_match &= (
                df["song_search"].str.contains(word, regex=False, na=False)
                | df["artist_search"].str.contains(word, regex=False, na=False)
            )
        filtered_df = df[word_match]

    filtered_df = filtered_df.head(100)
else:
    filtered_df = df.head(100)

if search_text.strip():
    st.caption(f"🔍 {len(filtered_df)} matching song(s) found")

if len(filtered_df) > 0:
    options = {}
    for index, row in filtered_df.iterrows():
        label = f"{row['song']} — {row['artist']}"
        if label not in options:
            options[label] = index

    selected_label = st.selectbox("🎧 Select a song", list(options.keys()))
    selected_index = options[selected_label]
else:
    st.warning("⚠️ No matching songs found.")
    st.info("Try entering only part of the song name or artist name.")
    selected_index = None

def get_recommendations(song_index, top_n=10):
    scores = similarity[song_index]
    sorted_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    recommendations = []
    for index in sorted_indices:
        if index == song_index:
            continue

        song_name = df.iloc[index]["song"]
        if not song_name.strip():
            continue

        recommendations.append({
            "song": song_name,
            "artist": df.iloc[index]["artist"],
            "score": float(scores[index])
        })

        if len(recommendations) >= top_n:
            break

    return recommendations

if selected_index is not None:
    if st.button("✨ Get Recommendations", use_container_width=True):
        st.session_state["recommendations"] = get_recommendations(
            selected_index, top_n=10
        )
        st.session_state["selected_song"] = df.iloc[selected_index]["song"]
        st.session_state["selected_artist"] = df.iloc[selected_index]["artist"]

if "recommendations" in st.session_state:
    st.divider()
    st.subheader("🎧 Selected Song")
    st.info(
        f"🎵 **{st.session_state['selected_song']}**  \\n"
        f"🎤 **Artist:** {st.session_state['selected_artist']}"
    )

    st.subheader("🎶 Recommended Songs")
    st.caption(
        "Recommendations are generated using lyric similarity "
        "with TF-IDF and Cosine Similarity."
    )

    recommendations = st.session_state["recommendations"]

    for i, recommendation in enumerate(recommendations, start=1):
        col1, col2 = st.columns([5, 2])

        with col1:
            st.markdown(f"### {i}. 🎵 {recommendation['song']}")
            st.write(f"🎤 **Artist:** {recommendation['artist']}")

        with col2:
            st.metric(
                "Similarity",
                f"{recommendation['score'] * 100:.2f}%"
            )

        if i != len(recommendations):
            st.divider()

st.divider()
st.caption("Built with Python • Pandas • Scikit-learn • Streamlit")
