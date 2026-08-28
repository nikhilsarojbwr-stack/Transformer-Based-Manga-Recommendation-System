"""
Manga Recommendation Engine
--------------------------
A dual‑mode (similar manga + natural language) semantic search app.
Powered by FAISS, BGE embeddings, and AniList data.
"""

import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from urllib.parse import urlparse
import time
from typing import Optional, List, Dict, Tuple, Set
import re

# ------------------------------
# Logging setup
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------
# Constants
# ------------------------------
PLACEHOLDER_IMAGE = "https://via.placeholder.com/300x450?text=No+Cover"
ANILIST_API_URL = "https://graphql.anilist.co"
ANILIST_QUERY = """
query ($search: String) {
  Media(search: $search, type: MANGA) {
    id
    title { english romaji }
    coverImage { extraLarge large medium color }
    description(asHtml: false)
    genres
    format
    popularity
    meanScore
    siteUrl
    externalLinks { url site }
  }
}
"""
DEFAULT_K = 20
MAX_RETRIES = 3
TIMEOUT = 15

# ------------------------------
# Custom CSS
# ------------------------------
def apply_custom_css():
    st.markdown("""
    <style>
        .main-header {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 20px 0px 10px 0px;
        }
        .main-header h1 {
            margin: 0;
            font-size: 2.8rem;
            background: linear-gradient(135deg, #ff6b6b, #ffd93d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stButton > button {
            background: linear-gradient(135deg, #6c5ce7, #a29bfe);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(108, 92, 231, 0.4);
        }
        .recommendation-card {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 5px solid #6c5ce7;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .info-tag {
            display: inline-block;
            background: #e9ecef;
            border-radius: 20px;
            padding: 0.1rem 0.8rem;
            font-size: 0.75rem;
            margin: 0.2rem 0.2rem;
        }
        .stProgress > div > div > div > div {
            background-color: #6c5ce7 !important;
        }
        .stAlert {
            border-left: 5px solid #ff6b6b;
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Utility: URL validation
# ------------------------------
def is_valid_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
        return all([parsed.scheme in ['http', 'https'], parsed.netloc])
    except:
        return False

# ------------------------------
# DataLoader: loads dataset, embeddings, FAISS
# ------------------------------
class DataLoader:
    @staticmethod
    @st.cache_resource(ttl=3600)
    def load_data() -> Tuple[pd.DataFrame, faiss.Index]:
        """Load CSV, embeddings, and build FAISS index with retries."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                df = pd.read_csv("semantic_manga_dataset.csv")
                required = ['english_title', 'romaji_title', 'semantic_text']
                missing = [c for c in required if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns: {missing}")

                embeddings = np.load("manga_embeddings.npy")
                if len(df) != len(embeddings):
                    raise ValueError("Dataset/embeddings size mismatch")

                faiss.normalize_L2(embeddings)
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)

                logger.info("Data loaded successfully.")
                return df, index
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {e}")
                if attempt == max_attempts - 1:
                    st.error("Could not load data after multiple attempts. Check files.")
                    st.stop()
                time.sleep(2 ** attempt)

    @staticmethod
    @st.cache_resource(ttl=3600)
    def get_genre_list(df: pd.DataFrame) -> List[str]:
        """Extract all unique genres from dataset."""
        genres = set()
        for g in df['genres'].dropna():
            genres.update([x.strip() for x in str(g).split(',')])
        return sorted(genres)

    @staticmethod
    @st.cache_resource(ttl=3600)
    def get_type_list(df: pd.DataFrame) -> List[str]:
        return sorted(df['format'].dropna().unique())

# ------------------------------
# ModelLoader: loads BGE embedding model
# ------------------------------
class ModelLoader:
    @staticmethod
    @st.cache_resource(ttl=3600)
    def load_model() -> SentenceTransformer:
        """Try primary model, then fallback options."""
        candidates = [
            "BAAI/bge-large-en-v1.5",
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L3-v2"
        ]
        for name in candidates:
            try:
                @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
                def _load():
                    return SentenceTransformer(name)
                model = _load()
                logger.info(f"Loaded model: {name}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
                continue
        st.error("No embedding model could be loaded.")
        st.stop()

# ------------------------------
# AniList API handler
# ------------------------------
class AniListClient:
    @staticmethod
    @st.cache_data(ttl=86400)  # cache for 1 day
    def fetch_media(title: str) -> Optional[Dict]:
        """Fetch manga metadata from AniList. Returns None on failure."""
        if not title or not isinstance(title, str):
            return None

        @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
        def _request():
            response = requests.post(
                ANILIST_API_URL,
                json={'query': ANILIST_QUERY, 'variables': {'search': title}},
                timeout=TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            if 'errors' in data:
                logger.warning(f"AniList error for {title}: {data['errors']}")
                return None
            return data.get('data', {}).get('Media')

        try:
            return _request()
        except Exception as e:
            logger.error(f"AniList fetch failed for {title}: {e}")
            return None

# ------------------------------
# Image display helper
# ------------------------------
def safe_image(cover_url: Optional[str], width: int = 200, caption: str = ""):
    """Display image with fallback to placeholder."""
    if cover_url and is_valid_url(cover_url):
        try:
            st.image(cover_url, width=width, caption=caption)
        except Exception:
            st.image(PLACEHOLDER_IMAGE, width=width, caption="No image")
    else:
        st.image(PLACEHOLDER_IMAGE, width=width, caption="No image")

# ------------------------------
# Core recommendation engine
# ------------------------------
def recommend_manga(
    query_text: str,                  # semantic text or user query
    df: pd.DataFrame,
    model: SentenceTransformer,
    index: faiss.Index,
    k: int = DEFAULT_K,
    exclude_title: Optional[str] = None,
    # Similar-mode filters
    ref_genres: Optional[Set[str]] = None,
    ref_type: Optional[str] = None,
    # Query-mode filters
    selected_genres: Optional[List[str]] = None,
    selected_type: Optional[str] = None,
    # Global toggles
    use_genre_filter: bool = True,
    use_type_filter: bool = True,
    allow_low_popularity: bool = True,
    sort_by: str = "rerank"
) -> Tuple[Optional[str], List[Dict]]:
    """
    Unified recommendation pipeline.

    Returns:
        (original_title, list_of_candidates)
        or (None, error_message) as string if no results.
    """
    try:
        # 1. Embed query
        query = "Represent this sentence for retrieval: " + query_text
        query_vec = model.encode([query], normalize_embeddings=True)

        # 2. FAISS search
        extra = 50
        max_candidates = min(k + extra, len(df))
        distances, indices = index.search(query_vec, max_candidates)

        candidates = []
        for i, dist in zip(indices[0], distances[0]):
            rec = df.iloc[i]
            rec_title = str(rec.get('english_title') or rec.get('romaji_title') or "Unknown Title")

            # skip invalid or excluded
            if not rec_title or rec_title.lower() == 'nan':
                continue
            if exclude_title and rec_title.lower() == exclude_title.lower():
                continue

            # --- Apply filters ---
            # Type filter
            if use_type_filter:
                if ref_type is not None:  # similar mode
                    if rec.get('format') != ref_type:
                        continue
                elif selected_type is not None:  # query mode
                    if rec.get('format') != selected_type:
                        continue

            # Genre filter
            if use_genre_filter:
                rec_genres = set(str(rec.get('genres', '')).split(','))
                if ref_genres is not None:  # similar mode: require overlap
                    if not (ref_genres & rec_genres):
                        continue
                elif selected_genres:  # query mode: require at least one selected genre
                    if not (set(selected_genres) & rec_genres):
                        continue

            # Popularity threshold
            popularity = rec.get('popularity', 0)
            if not allow_low_popularity and popularity < 1000:
                continue

            # --- Rerank score (only meaningful in similar mode) ---
            rerank_score = dist
            if ref_genres is not None and exclude_title:
                genre_overlap = len(ref_genres & rec_genres)
                title_overlap = len(
                    set(exclude_title.lower().split()) & set(rec_title.lower().split())
                )
                rerank_score = dist + 0.02 * genre_overlap + 0.01 * title_overlap

            candidates.append({
                'title': rec_title,
                'similarity': float(dist),
                'rerank_score': float(rerank_score),
                'popularity': int(popularity) if pd.notna(popularity) else 0,
                'description': rec.get('description', ''),
                'genres': rec.get('genres', ''),
                'format': rec.get('format', ''),
                'cover_url': rec.get('cover_url', ''),
                'anilist_id': rec.get('anilist_id', None)
            })

        if not candidates:
            return None, []  # no results

        # Sort
        if sort_by == "similarity":
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
        elif sort_by == "popularity":
            candidates.sort(key=lambda x: x['popularity'], reverse=True)
        else:  # rerank (default)
            candidates.sort(key=lambda x: (x['rerank_score'], x['popularity']), reverse=True)

        return None, candidates[:k]  # no original title (only needed for similar mode)

    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return None, []

# ------------------------------
# UI rendering
# ------------------------------
def render_sidebar(df: pd.DataFrame) -> Dict:
    """Build sidebar controls and return parameters."""
    st.sidebar.markdown("## ⚙️ Settings")

    mode = st.sidebar.radio(
        "Recommendation Mode",
        ["Similar Manga", "Search by Description"],
        index=0,
        help="Choose how you want to discover manga."
    )

    st.sidebar.divider()

    # Mode-specific inputs
    if mode == "Similar Manga":
        title_query = st.sidebar.selectbox(
            "Select a manga",
            options=sorted(df['english_title'].dropna().unique())
        )
        # Pre-fetch reference info (used for filtering)
        row = df[df['english_title'] == title_query].iloc[0]
        ref_genres = set(str(row.get('genres', '')).split(','))
        ref_type = row.get('format', '')
        selected_genres = None
        selected_type = None
        exclude_title = title_query
        query_text = row.get('semantic_text', '')
    else:  # Search by Description
        title_query = None
        ref_genres = None
        ref_type = None
        exclude_title = None
        query_text = st.sidebar.text_area(
            "Describe what you're looking for",
            placeholder="e.g., dark fantasy with psychological horror and demons",
            height=100
        )

        # If genre/type filters are active, show selectors
        st.sidebar.markdown("#### Filter options")
        selected_genres = st.sidebar.multiselect(
            "Genres to include",
            options=DataLoader.get_genre_list(df),
            default=[]
        ) if st.sidebar.checkbox("Filter by genre", value=True) else []
        selected_type = st.sidebar.selectbox(
            "Type",
            options=[""] + DataLoader.get_type_list(df),
            index=0
        ) if st.sidebar.checkbox("Filter by type", value=True) else None
        if selected_type == "":
            selected_type = None

    st.sidebar.divider()

    # Common controls
    k = st.sidebar.slider("Number of results", 5, 50, DEFAULT_K)
    use_genre_filter = st.sidebar.checkbox("Apply genre filter", value=True)
    use_type_filter = st.sidebar.checkbox("Apply type filter", value=True)
    allow_low_popularity = st.sidebar.checkbox("Include less popular titles", value=True)
    sort_by = st.sidebar.radio("Sort by", ["rerank", "similarity", "popularity"], index=0)

    return {
        'mode': mode,
        'title_query': title_query,
        'query_text': query_text,
        'exclude_title': exclude_title,
        'ref_genres': ref_genres,
        'ref_type': ref_type,
        'selected_genres': selected_genres,
        'selected_type': selected_type,
        'k': k,
        'use_genre_filter': use_genre_filter,
        'use_type_filter': use_type_filter,
        'allow_low_popularity': allow_low_popularity,
        'sort_by': sort_by
    }

def render_result_card(rec: Dict, anilist_data: Optional[Dict] = None):
    """Display a single recommendation as an expandable card."""
    with st.expander(f"📖 {rec['title']}  (score: {rec['rerank_score']:.3f})"):
        col1, col2 = st.columns([1, 3])

        with col1:
            # Determine best cover URL
            cover = None
            if anilist_data and anilist_data.get('coverImage'):
                for size in ['medium', 'large', 'extraLarge']:
                    if anilist_data['coverImage'].get(size):
                        cover = anilist_data['coverImage'][size]
                        break
            if not cover:
                cover = rec.get('cover_url')
            safe_image(cover, width=200, caption=rec['title'])

        with col2:
            # Basic info
            st.markdown(f"**Genres:** {rec.get('genres', 'N/A')}")
            st.markdown(f"**Type:** {rec.get('format', 'N/A')}")
            st.markdown(f"**Popularity:** {rec.get('popularity', 'N/A')}")

            # AniList extras
            if anilist_data:
                if anilist_data.get('meanScore'):
                    st.markdown(f"**AniList Score:** ⭐ {anilist_data['meanScore']}/100")
                if anilist_data.get('siteUrl'):
                    st.markdown(f"[🔗 View on AniList]({anilist_data['siteUrl']})")
                if anilist_data.get('externalLinks'):
                    links = [f"[{link['site']}]({link['url']})" for link in anilist_data['externalLinks'][:3]]
                    if links:
                        st.markdown("**Official links:** " + " | ".join(links))

            # Description
            desc = anilist_data.get('description') if anilist_data else None
            if not desc:
                desc = rec.get('description', '')
            if desc:
                st.caption(desc[:500] + "..." if len(desc) > 500 else desc)
            else:
                st.caption("No description available.")

# ------------------------------
# Main app
# ------------------------------
def main():
    st.set_page_config(
        page_title="Manga Recommender",
        layout="wide",
        page_icon="📚",
        initial_sidebar_state="expanded"
    )
    apply_custom_css()

    # Header with logo-like design
    st.markdown("""
    <div class="main-header">
        <h1>📚 Manga Recommender</h1>
        <span style="font-size:1.2rem; color:#6c5ce7;">Powered by AI & FAISS</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Find your next favourite manga using semantic similarity or natural language.")

    # Load resources
    with st.spinner("Loading data and AI model..."):
        df, index = DataLoader.load_data()
        model = ModelLoader.load_model()

    # Sidebar parameters
    params = render_sidebar(df)

    # Main search button
    if st.sidebar.button("🔍 Find Recommendations", type="primary"):

        # Validate query
        if params['mode'] == "Search by Description" and not params['query_text'].strip():
            st.warning("Please enter a description.")
            return

        # Call the unified recommender
        with st.spinner("Searching..."):
            original_title, candidates = recommend_manga(
                query_text=params['query_text'],
                df=df,
                model=model,
                index=index,
                k=params['k'],
                exclude_title=params['exclude_title'],
                ref_genres=params['ref_genres'],
                ref_type=params['ref_type'],
                selected_genres=params['selected_genres'],
                selected_type=params['selected_type'],
                use_genre_filter=params['use_genre_filter'],
                use_type_filter=params['use_type_filter'],
                allow_low_popularity=params['allow_low_popularity'],
                sort_by=params['sort_by']
            )

        if not candidates:
            st.warning("No recommendations found. Try loosening filters or changing your query.")
            return

        # Show source info (if similar mode)
        if params['mode'] == "Similar Manga" and params['title_query']:
            src_title = params['title_query']
            src_data = AniListClient.fetch_media(src_title)
            st.success(f"Based on **{src_title}** – found {len(candidates)} recommendations.")
            if src_data:
                col1, col2 = st.columns([1, 4])
                with col1:
                    cover = src_data.get('coverImage', {}).get('large')
                    safe_image(cover, width=150, caption=src_title)
                with col2:
                    st.markdown(f"### {src_title}")
                    if src_data.get('meanScore'):
                        st.markdown(f"**AniList Score:** ⭐ {src_data['meanScore']}/100")
                    if src_data.get('siteUrl'):
                        st.markdown(f"[View on AniList]({src_data['siteUrl']})")
                    if src_data.get('description'):
                        st.caption(src_data['description'][:300] + "...")
        else:
            st.success(f"Found {len(candidates)} recommendations for your query.")

        st.divider()

        # Display results with progress bar
        progress = st.progress(0)
        for i, rec in enumerate(candidates, 1):
            # Fetch AniList data for each (cached)
            anilist_data = AniListClient.fetch_media(rec['title'])
            render_result_card(rec, anilist_data)
            progress.progress(i / len(candidates))
        progress.empty()

# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    main()