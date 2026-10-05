import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (Clean, Colorful & Modern)
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Main container styling */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1E88E5, #7E57C2, #E91E63);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #546E7A;
        margin-bottom: 1.8rem;
    }
    /* Card design */
    .book-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
        min-height: 220px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    .book-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        border-color: #90CAF9;
    }
    .badge-genre {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #E3F2FD;
        color: #1565C0;
        margin-bottom: 0.6rem;
    }
    .badge-rating {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #FFF9C4;
        color: #F57F17;
    }
    .selected-card {
        background: linear-gradient(135deg, #E3F2FD, #EDE7F6);
        border: 1px solid #BBDEFB;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data & Model Loader with Robust Error Handling
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_data():
    """
    Loads dataset and model artifacts.
    If the saved pickle is missing, dynamically trains TF-IDF and computes Cosine Similarity.
    """
    dataset_path = os.path.join("data", "books.csv")
    model_path = os.path.join("model", "recommendation_model.pkl")

    if not os.path.exists(dataset_path):
        return None, None, None, f"⚠️ Unable to load the book dataset. Missing file: `{dataset_path}`."

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return None, None, None, f"⚠️ Error reading books dataset: {str(e)}"

    if df.empty:
        return None, None, None, "⚠️ The dataset is empty."

    required_cols = {"book_id", "title", "author", "genre", "description", "rating", "year"}
    if not required_cols.issubset(set(df.columns)):
        return None, None, None, "⚠️ Dataset is missing required columns."

    # Preprocessing
    df.fillna("", inplace=True)

    # Attempt to load serialized model if available
    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            if "df" in model_data and "similarity_matrix" in model_data:
                return model_data["df"], model_data.get("tfidf"), model_data["similarity_matrix"], None
        except Exception:
            pass  # Fall back to on-the-fly computation

    # Dynamic fallback computation
    try:
        df["combined_features"] = df["title"] + " " + df["author"] + " " + df["genre"] + " " + df["description"]
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(df["combined_features"])
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Cache model to disk for future fast loads
        try:
            os.makedirs("model", exist_ok=True)
            joblib.dump({"df": df, "tfidf": tfidf, "similarity_matrix": similarity_matrix}, model_path)
        except Exception:
            pass

        return df, tfidf, similarity_matrix, None
    except Exception as e:
        return None, None, None, f"⚠️ Model loading error: {str(e)}"

# ---------------------------------------------------------
# Recommendation Function
# ---------------------------------------------------------
def recommend_books(book_title, df, similarity_matrix, n=5):
    """
    Finds top-N similar books using precomputed cosine similarity matrix.
    Excludes the input book itself.
    """
    if book_title not in df["title"].values:
        return None, f"⚠️ Book '{book_title}' not found in the catalog."

    idx = df[df["title"] == book_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the selected book itself
    filtered_scores = [item for item in sim_scores if item[0] != idx][:n]
    book_indices = [item[0] for item in filtered_scores]

    results = df.iloc[book_indices].copy()
    results["similarity_score"] = [item[1] for item in filtered_scores]
    return results, None

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 📚 About the Project")
    st.info(
        "This **Book Recommendation System** uses **Content-Based Filtering** "
        "with **TF-IDF Vectorization** and **Cosine Similarity** to recommend "
        "books similar to the selected book."
    )
    st.markdown("---")
    st.markdown("### 🤖 Machine Learning Mini Project")
    st.markdown("""
    **Core Pipeline:**
    - 📚 **Dataset:** Custom Curated Catalog
    - 🧹 **Preprocessing:** Text Normalization
    - 📝 **Features:** Metadata Combination
    - 🔢 **TF-IDF:** Numerical Vectorization
    - 📐 **Similarity:** Cosine Similarity Metric
    - 🎯 **Output:** Top-N Book Matches
    """)
    st.markdown("---")
    st.caption("College ML Mini Project • Streamlit UI")

# ---------------------------------------------------------
# Main Application Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">📚 Book Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Discover your next favorite book! Find books similar to the ones you love.</div>', unsafe_allow_html=True)

# Load resources
df, tfidf, similarity_matrix, error_msg = load_model_and_data()

if error_msg:
    st.error(error_msg)
    st.stop()

# ---------------------------------------------------------
# Interactive Selection Controls
# ---------------------------------------------------------
col_input, col_slider = st.columns([3, 2])

book_titles = sorted(df["title"].tolist())

with col_input:
    selected_book = st.selectbox(
        "📖 Select a Book",
        options=book_titles,
        index=book_titles.index("The Hobbit") if "The Hobbit" in book_titles else 0,
        help="Choose any book from our library to get similar recommendations"
    )

with col_slider:
    num_recommendations = st.slider(
        "🔢 Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help="Select between 3 and 10 recommendations"
    )

# Selected Book Overview Card
if selected_book:
    selected_info = df[df["title"] == selected_book].iloc[0]
    st.markdown(f"""
        <div class="selected-card">
            <h4 style="margin-top:0; color: #1E3A8A;">🎯 Selected Book: <strong>{selected_info['title']}</strong></h4>
            <p style="margin: 4px 0;">✍️ <strong>Author:</strong> {selected_info['author']} &nbsp;|&nbsp; 🏷️ <strong>Genre:</strong> {selected_info['genre']} &nbsp;|&nbsp; ⭐ <strong>Rating:</strong> {selected_info['rating']} / 5.0 &nbsp;|&nbsp; 📅 <strong>Year:</strong> {selected_info['year']}</p>
            <p style="margin: 6px 0 0 0; color: #374151; font-size: 0.95rem;"><em>"{selected_info['description']}"</em></p>
        </div>
    """, unsafe_allow_html=True)

recommend_clicked = st.button("🔍 Recommend Books", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Recommendations Presentation
# ---------------------------------------------------------
if recommend_clicked:
    if not selected_book:
        st.warning("⚠️ Please select a valid book.")
    else:
        results, err = recommend_books(selected_book, df, similarity_matrix, n=num_recommendations)
        if err:
            st.error(err)
        elif results is not None and not results.empty:
            st.markdown(f"### 🌟 Top {len(results)} Recommendations for *{selected_book}*:")

            # Display cards in a multi-column responsive layout
            grid_cols = st.columns(3)

            for i, (_, row) in enumerate(results.iterrows()):
                col = grid_cols[i % 3]
                with col:
                    st.markdown(f"""
                        <div class="book-card">
                            <span class="badge-genre">{row['genre']}</span>
                            <span class="badge-rating">⭐ {row['rating']}</span>
                            <h4 style="margin: 0.4rem 0 0.2rem 0; color: #1F2937; line-height: 1.3;">📖 {row['title']}</h4>
                            <p style="margin: 0 0 0.5rem 0; color: #4B5563; font-size: 0.92rem;">✍️ {row['author']}</p>
                            <p style="margin: 0; color: #6B7280; font-size: 0.85rem;">📅 Published: {row['year']}</p>
                            <p style="margin: 0.3rem 0 0 0; color: #0284C7; font-size: 0.85rem; font-weight: 600;">Match Score: {round(row['similarity_score'] * 100, 1)}%</p>
                        </div>
                    """, unsafe_allow_html=True)

                    with st.expander("📖 About this book"):
                        st.write(row['description'])
        else:
            st.info("No recommendations found.")
else:
    st.markdown("---")
    st.caption("💡 Select a book above and click **'🔍 Recommend Books'** to discover matches.")
