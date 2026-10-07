import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


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
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
    <style>

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(
            120deg,
            #1E88E5,
            #7E57C2,
            #E91E63
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .sub-title {
        font-size: 1.15rem;
        color: #546E7A;
        margin-bottom: 1.8rem;
    }

    .book-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
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
        background: linear-gradient(
            135deg,
            #E3F2FD,
            #EDE7F6
        );
        border: 1px solid #BBDEFB;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }

    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Dataset and Train KNN Model
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_data():

    dataset_path = os.path.join("data", "books.csv")

    if not os.path.exists(dataset_path):
        return None, None, None, None, (
            f"⚠️ Unable to load the book dataset. "
            f"Missing file: `{dataset_path}`."
        )

    try:
        df = pd.read_csv(dataset_path)

    except Exception as e:
        return None, None, None, None, (
            f"⚠️ Error reading books dataset: {str(e)}"
        )

    if df.empty:
        return None, None, None, None, (
            "⚠️ The dataset is empty."
        )

    # Required columns
    required_cols = {
        "book_id",
        "title",
        "author",
        "genre",
        "description",
        "rating",
        "year"
    }

    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)

        return None, None, None, None, (
            f"⚠️ Dataset is missing required columns: {missing}"
        )

    # -----------------------------------------------------
    # Data Preprocessing
    # -----------------------------------------------------
    df = df.copy()

    text_columns = [
        "title",
        "author",
        "genre",
        "description"
    ]

    for col in text_columns:
        df[col] = df[col].fillna("").astype(str)

    # -----------------------------------------------------
    # Combine Book Features
    # -----------------------------------------------------
    df["combined_features"] = (
        df["title"] + " " +
        df["author"] + " " +
        df["genre"] + " " +
        df["description"]
    )

    # -----------------------------------------------------
    # TF-IDF Vectorization
    # -----------------------------------------------------
    tfidf = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = tfidf.fit_transform(
        df["combined_features"]
    )

    # -----------------------------------------------------
    # KNN MODEL
    # -----------------------------------------------------
    knn_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    knn_model.fit(tfidf_matrix)

    # -----------------------------------------------------
    # Save Model
    # -----------------------------------------------------
    try:
        os.makedirs("model", exist_ok=True)

        joblib.dump(
            {
                "df": df,
                "tfidf": tfidf,
                "knn_model": knn_model
            },
            "model/knn_recommendation_model.pkl"
        )

    except Exception:
        pass

    return (
        df,
        tfidf,
        tfidf_matrix,
        knn_model,
        None
    )


# ---------------------------------------------------------
# KNN Recommendation Function
# ---------------------------------------------------------
def recommend_books(
    book_title,
    df,
    tfidf,
    tfidf_matrix,
    knn_model,
    n=5
):

    if book_title not in df["title"].values:

        return None, (
            f"⚠️ Book '{book_title}' "
            f"not found in the catalog."
        )

    # Find selected book index
    book_index = df[
        df["title"] == book_title
    ].index[0]

    # Get TF-IDF vector of selected book
    book_vector = tfidf_matrix[
        book_index
    ]

    # -----------------------------------------------------
    # KNN Prediction
    # -----------------------------------------------------
    distances, indices = knn_model.kneighbors(
        book_vector,
        n_neighbors=min(n + 1, len(df))
    )

    results = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        # Skip selected book itself
        if index == book_index:
            continue

        similarity = 1 - distance

        results.append(
            {
                "index": index,
                "similarity_score": similarity
            }
        )

        if len(results) == n:
            break

    if not results:
        return None, "⚠️ No recommendations found."

    # -----------------------------------------------------
    # Create Result DataFrame
    # -----------------------------------------------------
    book_indices = [
        item["index"]
        for item in results
    ]

    result_df = df.iloc[
        book_indices
    ].copy()

    result_df["similarity_score"] = [
        item["similarity_score"]
        for item in results
    ]

    return result_df, None


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:

    st.markdown("## 📚 About the Project")

    st.info(
        "This **Book Recommendation System** uses "
        "**Machine Learning with K-Nearest Neighbors (KNN)** "
        "and **TF-IDF Vectorization** to recommend books "
        "similar to the selected book."
    )

    st.markdown("---")

    st.markdown("### 🤖 Machine Learning Mini Project")

    st.markdown("""
    **ML Pipeline:**

    - 📚 **Dataset:** Book Catalog
    - 🧹 **Preprocessing:** Text Cleaning
    - 📝 **Features:** Title, Author, Genre, Description
    - 🔢 **TF-IDF:** Feature Vectorization
    - 🤖 **Algorithm:** K-Nearest Neighbors (KNN)
    - 📐 **Distance:** Cosine Distance
    - 🎯 **Output:** Top-N Similar Books
    """)

    st.markdown("---")

    st.markdown("### 🧠 Algorithm")

    st.write(
        """
        **KNN** finds books that are closest to the
        selected book based on their TF-IDF feature
        vectors.
        """
    )

    st.markdown("---")

    st.caption(
        "College ML Mini Project • Streamlit UI"
    )


# ---------------------------------------------------------
# Main Application Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">'
    '📚 Book Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Discover your next favorite book using '
    'Machine Learning and KNN.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Load Model and Data
# ---------------------------------------------------------
df, tfidf, tfidf_matrix, knn_model, error_msg = (
    load_model_and_data()
)

if error_msg:

    st.error(error_msg)

    st.stop()


# ---------------------------------------------------------
# Interactive Selection Controls
# ---------------------------------------------------------
col_input, col_slider = st.columns([3, 2])

book_titles = sorted(
    df["title"].tolist()
)


with col_input:

    default_index = (
        book_titles.index("The Hobbit")
        if "The Hobbit" in book_titles
        else 0
    )

    selected_book = st.selectbox(
        "📖 Select a Book",
        options=book_titles,
        index=default_index,
        help=(
            "Choose a book from the library "
            "to get similar recommendations."
        )
    )


with col_slider:

    num_recommendations = st.slider(
        "🔢 Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help=(
            "Select between 3 and 10 recommendations."
        )
    )


# ---------------------------------------------------------
# Selected Book Overview
# ---------------------------------------------------------
if selected_book:

    selected_info = df[
        df["title"] == selected_book
    ].iloc[0]

    st.markdown(
        f"""
        <div class="selected-card">

            <h4 style="
                margin-top:0;
                color:#1E3A8A;
            ">
                🎯 Selected Book:
                <strong>{selected_info['title']}</strong>
            </h4>

            <p style="margin:4px 0;">
                ✍️ <strong>Author:</strong>
                {selected_info['author']}
                &nbsp;|&nbsp;

                🏷️ <strong>Genre:</strong>
                {selected_info['genre']}
                &nbsp;|&nbsp;

                ⭐ <strong>Rating:</strong>
                {selected_info['rating']} / 5.0
                &nbsp;|&nbsp;

                📅 <strong>Year:</strong>
                {selected_info['year']}
            </p>

            <p style="
                margin:6px 0 0 0;
                color:#374151;
                font-size:0.95rem;
            ">
                <em>
                    "{selected_info['description']}"
                </em>
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# Recommendation Button
# ---------------------------------------------------------
recommend_clicked = st.button(
    "🔍 Recommend Books",
    type="primary",
    use_container_width=True
)


# ---------------------------------------------------------
# Display Recommendations
# ---------------------------------------------------------
if recommend_clicked:

    if not selected_book:

        st.warning(
            "⚠️ Please select a valid book."
        )

    else:

        results, err = recommend_books(
            selected_book,
            df,
            tfidf,
            tfidf_matrix,
            knn_model,
            n=num_recommendations
        )

        if err:

            st.error(err)

        elif (
            results is not None
            and not results.empty
        ):

            st.markdown(
                f"""
                ### 🌟 Top {len(results)}
                Recommendations for
                *{selected_book}*
                """
            )

            # -------------------------------------------------
            # Recommendation Cards
            # -------------------------------------------------
            grid_cols = st.columns(3)

            for i, (_, row) in enumerate(
                results.iterrows()
            ):

                col = grid_cols[
                    i % 3
                ]

                with col:

                    st.markdown(
                        f"""
                        <div class="book-card">

                            <span class="badge-genre">
                                {row['genre']}
                            </span>

                            <span class="badge-rating">
                                ⭐ {row['rating']}
                            </span>

                            <h4 style="
                                margin:0.4rem 0 0.2rem 0;
                                color:#1F2937;
                                line-height:1.3;
                            ">
                                📖 {row['title']}
                            </h4>

                            <p style="
                                margin:0 0 0.5rem 0;
                                color:#4B5563;
                                font-size:0.92rem;
                            ">
                                ✍️ {row['author']}
                            </p>

                            <p style="
                                margin:0;
                                color:#6B7280;
                                font-size:0.85rem;
                            ">
                                📅 Published:
                                {row['year']}
                            </p>

                            <p style="
                                margin:0.3rem 0 0 0;
                                color:#0284C7;
                                font-size:0.85rem;
                                font-weight:600;
                            ">
                                KNN Similarity:
                                {round(
                                    row['similarity_score'] * 100,
                                    1
                                )}%
                            </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    with st.expander(
                        "📖 About this book"
                    ):

                        st.write(
                            row["description"]
                        )

        else:

            st.info(
                "No recommendations found."
            )


# ---------------------------------------------------------
# Footer Message
# ---------------------------------------------------------
else:

    st.markdown("---")

    st.caption(
        "💡 Select a book above and click "
        "**'🔍 Recommend Books'** to discover "
        "similar books using KNN."
)
