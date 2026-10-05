# Book Recommendation System

## Project Overview
The **Book Recommendation System** is a machine learning mini-project designed to help readers discover new books matching their tastes. Using **Content-Based Filtering**, the system analyzes the metadata (title, author, genre, and detailed descriptions) of a book selected by the user and recommends the most topically similar books in the catalog.

This project was built for a college Machine Learning subject mini-project. It is simple, clean, functional, colorful, and easy to explain during a presentation or viva examination.

## Objective
The primary objective of this project is to implement an end-to-end Machine Learning pipeline that:
- Allows the user to select a book from a library catalog.
- Analyzes the content and metadata of the selected book.
- Computes mathematical similarity against all other books.
- Recommends the top-N most similar books.
- Presents the recommendations through a colorful, interactive, and responsive Streamlit user interface.

## Technologies Used
- **Python**: Core programming language.
- **Pandas**: Data manipulation, filtering, and analysis.
- **NumPy**: Numerical operations and vector calculations.
- **Scikit-learn**:
  - `TfidfVectorizer` for text numerical transformation.
  - `cosine_similarity` for measuring similarity between book vectors.
- **Streamlit**: Modern interactive web UI.
- **Joblib**: Lightweight serialization for saving and loading the model pipeline.

## Dataset
The dataset is a **custom-created dataset included in the project** stored at `data/books.csv`. 
- No external APIs or manual downloads are needed.
- Contains 60 carefully curated books across 15 popular genres:
  - Fantasy, Science Fiction, Mystery, Thriller, Fiction, Romance, Adventure, Historical Fiction, Self-Help, Psychology, Business, Technology, Biography, Philosophy, and Young Adult.
- Each record has 7 standard attributes: `book_id`, `title`, `author`, `genre`, `description`, `rating`, and `year`.
- The descriptions are topically rich so that the recommendation algorithm discovers natural, sensible similarities between related books.

## Machine Learning Approach
This system uses **Content-Based Filtering**:
1. **Feature Engineering**: A composite text string is created by concatenating:
   ```python
   combined_features = title + author + genre + description
   ```
2. **TF-IDF Vectorization**: 
   - **Term Frequency (TF)** measures how often a word occurs in a specific book.
   - **Inverse Document Frequency (IDF)** diminishes the weight of common stopwords and highlights distinctive keywords (e.g., *hobbit*, *dystopian*, *detective*).
3. **Cosine Similarity**:
   Calculates the cosine of the angle between two book feature vectors in high-dimensional space:
   $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
   Values close to 1.0 indicate high topical similarity.

## Project Workflow
```text
Custom Book Dataset
        ↓
Data Preprocessing
        ↓
Feature Combination
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
Similar Books
        ↓
Top-N Recommendations
        ↓
Streamlit UI
```

## Project Structure
```text
Book-Recommendation-System/
│
├── app.py                         # Streamlit web application
├── create_dataset.py              # Custom dataset generator
├── requirements.txt               # Dependencies list
├── README.md                      # Documentation & viva reference
│
├── data/
│   └── books.csv                  # Curated custom book dataset
│
├── notebooks/
│   └── book_recommendation.ipynb  # Step-by-step ML workflow notebook
│
└── model/
    └── recommendation_model.pkl   # Serialized model artifact
```

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset (Optional - already generated)
```bash
python create_dataset.py
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the interactive app.

## Features
- 📚 **Custom book dataset**: 60 diverse, realistic books across 15 genres.
- 🔍 **Book selection**: Dropdown selector automatically populated from the dataset.
- 🤖 **Content-based recommendation**: Pure algorithmic recommendation without hardcoding.
- 🧠 **TF-IDF vectorization**: Meaningful natural language representation.
- 📐 **Cosine similarity**: Mathematically rigorous ranking.
- ⭐ **Book ratings & year**: Displayed clearly on each recommendation card.
- 🏷️ **Genre information**: Colorful genre pill badges.
- 📖 **Book descriptions**: Interactive expandable view for every book.
- 🔢 **Adjustable recommendation count**: Slider to select 3 to 10 recommendations.
- 🎨 **Colorful Streamlit UI**: Modern gradient header, cards, and hover effects.

## Future Enhancements
- Incorporate hybrid filtering by factoring in user ratings and collaborative filtering.
- Include book cover image URLs for richer visual presentation.
- Add multi-criteria filtering by genre, minimum rating, or publication era.
