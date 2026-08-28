import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from PIL import Image
import io
import time
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
TIMEOUT = 15
PLACEHOLDER_IMAGE = "https://via.placeholder.com/300x450?text=No+Cover+Available"

# AniList API GraphQL query
ANILIST_QUERY = """
query ($search: String) {
  Media(search: $search, type: MANGA) {
    id
    title {
      english
      romaji
    }
    coverImage {
      extraLarge
      large
      medium
      color
    }
    description(asHtml: false)
    genres
    format
    popularity
    meanScore
    siteUrl
    externalLinks {
      url
      site
    }
  }
}
"""

class DataLoader:
    """Handles all data loading operations with error recovery"""
    
    @staticmethod
    @st.cache_resource(ttl=3600)
    def load_data():
        """Load dataset, embeddings, and build FAISS index with error handling"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Load dataset
                df = pd.read_csv("semantic_manga_dataset.csv")
                
                # Validate essential columns
                required_columns = ['english_title', 'romaji_title', 'semantic_text']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
                
                # Load embeddings
                embeddings = np.load("manga_embeddings.npy")
                
                # Validate shapes match
                if len(df) != len(embeddings):
                    raise ValueError("Dataset and embeddings size mismatch")
                
                # Build FAISS index
                faiss.normalize_L2(embeddings)
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                
                logger.info("Data loaded successfully")
                return df, index
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    st.error("Failed to load dataset after multiple attempts. Please check your data files.")
                    st.stop()
                time.sleep(retry_delay)
                retry_delay *= 2

class ModelLoader:
    """Handles model loading with fallback strategies"""
    
    @staticmethod
    @st.cache_resource(ttl=3600)
    def load_model():
        """Load SentenceTransformer model with multiple fallback options"""
        model_names = [
            "BAAI/bge-large-en-v1.5",  # Primary model
            "all-MiniLM-L6-v2",        # Smaller fallback
            "paraphrase-MiniLM-L3-v2"   # Lightweight fallback
        ]
        
        for model_name in model_names:
            try:
                @retry(stop=stop_after_attempt(3), 
                      wait=wait_exponential(multiplier=1, min=4, max=10))
                def attempt_load():
                    return SentenceTransformer(model_name)
                
                model = attempt_load()
                logger.info(f"Successfully loaded model: {model_name}")
                return model
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
                continue
        
        st.error("Failed to load any embedding model. Please check your internet connection.")
        st.stop()

class APIHandler:
    """Handles all API communications with robust error handling"""
    
    @staticmethod
    @retry(stop=stop_after_attempt(MAX_RETRIES), 
          wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_anilist_data(title):
        """Fetch manga data from AniList API with comprehensive error handling"""
        try:
            if not title or not isinstance(title, str):
                raise ValueError("Invalid title provided")
                
            variables = {'search': title}
            response = requests.post(
                'https://graphql.anilist.co',
                json={'query': ANILIST_QUERY, 'variables': variables},
                timeout=TIMEOUT
            )
            
            response.raise_for_status()
            
            data = response.json()
            if 'errors' in data:
                error_msg = data['errors'][0].get('message', 'Unknown GraphQL error')
                raise ValueError(f"AniList API error: {error_msg}")
                
            return data.get('data', {}).get('Media')
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching data for {title}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {title}: {str(e)}")
            return None

class ImageHandler:
    """Handles all image operations with safety checks"""
    
    @staticmethod
    def is_valid_url(url):
        """Check if a URL is valid and properly formatted"""
        try:
            if not url or not isinstance(url, str):
                return False
                
            parsed = urlparse(url)
            return all([parsed.scheme in ['http', 'https'], parsed.netloc])
        except:
            return False
    
    @staticmethod
    def safe_image_display(image_url, width=200, alt_text="Cover Image"):
        """Display an image with multiple layers of fallback"""
        try:
            # Validate URL first
            if not ImageHandler.is_valid_url(image_url):
                raise ValueError("Invalid image URL")
                
            # Try to display the image
            st.image(image_url, width=width, caption=alt_text)
            
        except Exception as e:
            logger.warning(f"Image display failed: {str(e)}. Using placeholder.")
            st.image(PLACEHOLDER_IMAGE, width=width, caption="Image not available")

class Recommender:
    """Handles all recommendation logic"""
    
    @staticmethod
    def recommend_manga(title, df, model, index, k=20, 
                      genre_filter=True, type_filter=True, 
                      allow_low_popularity=True, sort_by="rerank"):
        """Generate manga recommendations with comprehensive error handling"""
        try:
            # Validate inputs
            if not title or not isinstance(title, str):
                raise ValueError("Invalid title provided")
                
            # Find the manga in dataset
            row = df[df['english_title'].str.lower() == title.lower()]
            if row.empty:
                row = df[df['romaji_title'].str.lower() == title.lower()]
                if row.empty:
                    return None, f"Title '{title}' not found in dataset."

            row = row.iloc[0]
            semantic_text = row.get('semantic_text', '')
            if not semantic_text:
                raise ValueError("No semantic text available for this title")
                
            original_title = row.get('english_title') or row.get('romaji_title') or "Unknown Title"
            original_type = row.get('format', 'MANGA')
            original_genres = set(str(row.get('genres', '')).split(','))

            # Embed query with safety checks
            try:
                query = "Represent this sentence for retrieval: " + semantic_text
                query_vec = model.encode([query], normalize_embeddings=True)
            except Exception as e:
                raise ValueError(f"Embedding failed: {str(e)}")

            # Get initial candidates
            try:
                distances, indices = index.search(query_vec, min(k + 50, len(df)))
            except Exception as e:
                raise ValueError(f"FAISS search failed: {str(e)}")

            candidates = []
            for i, dist in zip(indices[0], distances[0]):
                try:
                    rec = df.iloc[i]
                    rec_title = str(rec.get('english_title') or rec.get('romaji_title') or "Unknown Title")

                    # Skip invalid or duplicate titles
                    if not rec_title or rec_title.lower() == title.lower() or rec_title.lower() == 'nan':
                        continue

                    # Apply filters
                    if type_filter and rec.get('format') != original_type:
                        continue

                    rec_genres = set(str(rec.get('genres', '')).split(','))
                    if genre_filter and not original_genres & rec_genres:
                        continue

                    popularity = rec.get('popularity', 0)
                    if not allow_low_popularity and popularity < 1000:
                        continue

                    # Calculate rerank score
                    genre_overlap = len(original_genres & rec_genres)
                    title_overlap = len(set(original_title.lower().split()) & set(rec_title.lower().split()))
                    rerank_score = dist + 0.02 * genre_overlap + 0.01 * title_overlap

                    # Validate cover URL
                    cover_url = rec.get('cover_url', '')
                    if not ImageHandler.is_valid_url(cover_url):
                        cover_url = ''

                    candidates.append({
                        'title': rec_title,
                        'similarity': dist,
                        'rerank_score': rerank_score,
                        'popularity': popularity,
                        'description': rec.get('description', 'N/A'),
                        'genres': rec.get('genres', 'N/A'),
                        'format': rec.get('format', 'N/A'),
                        'cover_url': cover_url,
                        'anilist_id': rec.get('anilist_id')
                    })
                except Exception as e:
                    logger.warning(f"Skipping invalid recommendation entry: {str(e)}")
                    continue

            # Sort results
            if not candidates:
                return None, "No valid recommendations found after filtering"
                
            if sort_by == "similarity":
                candidates.sort(key=lambda x: x['similarity'], reverse=True)
            elif sort_by == "popularity":
                candidates.sort(key=lambda x: x['popularity'], reverse=True)
            else:  # rerank
                candidates.sort(key=lambda x: (x['rerank_score'], x['popularity']), reverse=True)

            return original_title, candidates[:k]
            
        except Exception as e:
            logger.error(f"Recommendation error: {str(e)}")
            return None, f"❌ Error generating recommendations: {str(e)}"

class UI:
    """Handles all user interface components"""
    
    @staticmethod
    def display_recommendation(rec, anilist_data=None):
        """Display a recommendation card with robust error handling"""
        try:
            with st.expander(f"{rec['title']} (Score: {rec['rerank_score']:.3f})"):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Try AniList image first
                    cover_url = None
                    if anilist_data and anilist_data.get('coverImage'):
                        for size in ['medium', 'large', 'extraLarge']:
                            if anilist_data['coverImage'].get(size):
                                cover_url = anilist_data['coverImage'][size]
                                break
                    
                    # Fallback to dataset image
                    if not cover_url and rec.get('cover_url'):
                        cover_url = rec['cover_url']
                    
                    ImageHandler.safe_image_display(cover_url, 200, rec['title'])
                
                with col2:
                    # Basic info
                    st.markdown(f"""
                    **Genres:** {rec.get('genres', 'N/A')}  
                    **Type:** {rec.get('format', 'N/A')}  
                    **Popularity:** {rec.get('popularity', 'N/A')}
                    """)
                    
                    # AniList-specific info if available
                    if anilist_data:
                        st.markdown(f"**AniList Score:** {anilist_data.get('meanScore', 'N/A')}")
                        if anilist_data.get('siteUrl'):
                            st.markdown(f"**[View on AniList]({anilist_data['siteUrl']})**")
                    
                    # Description with fallbacks
                    desc = ''
                    if anilist_data and anilist_data.get('description'):
                        desc = anilist_data['description']
                    elif rec.get('description'):
                        desc = rec['description']
                    
                    if desc:
                        st.caption(desc[:500] + "..." if len(desc) > 500 else desc)
                    else:
                        st.caption("No description available")
                    
                    # External links if available
                    if anilist_data and anilist_data.get('externalLinks'):
                        st.markdown("**Official Links:**")
                        for link in anilist_data['externalLinks']:
                            try:
                                st.markdown(f"- [{link['site']}]({link['url']})")
                            except:
                                continue
        except Exception as e:
            logger.error(f"Failed to display recommendation: {str(e)}")
            st.error("Couldn't display this recommendation properly")

    @staticmethod
    def setup_sidebar(df):
        """Create the sidebar controls"""
        st.sidebar.header("Search Settings")
        
        try:
            titles = sorted(df['english_title'].dropna().unique())
            if not titles:
                raise ValueError("No valid titles found in dataset")
                
            title_query = st.sidebar.selectbox(
                "Select a manga title",
                options=titles,
                index=0
            )
            
            k = st.sidebar.slider("Number of recommendations", 5, 50, 20)
            genre_filter = st.sidebar.checkbox("Filter by genre", value=True)
            type_filter = st.sidebar.checkbox("Filter by type", value=True)
            allow_low_popularity = st.sidebar.checkbox("Include less popular titles", value=True)
            sort_by = st.sidebar.radio("Sort by", ["rerank", "similarity", "popularity"], index=0)
            
            return {
                'title_query': title_query,
                'k': k,
                'genre_filter': genre_filter,
                'type_filter': type_filter,
                'allow_low_popularity': allow_low_popularity,
                'sort_by': sort_by
            }
            
        except Exception as e:
            logger.error(f"Sidebar setup failed: {str(e)}")
            st.sidebar.error("Failed to initialize search controls")
            st.stop()

def main():
    st.set_page_config(
        page_title="Manga Recommender", 
        layout="wide",
        page_icon="📚"
    )
    
    # Add custom CSS for better error visibility
    st.markdown("""
    <style>
        .stAlert {
            border-left: 5px solid #ff4b4b;
        }
        .recommendation-card {
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: #f0f2f6;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 Manga Recommendation Engine")
    st.markdown("Discover new manga based on semantic similarity and popularity")
    
    # Initialize session state for error tracking
    if 'errors' not in st.session_state:
        st.session_state.errors = []
    
    try:
        # Load data and model
        with st.spinner("Loading data and AI model..."):
            df, index = DataLoader.load_data()
            model = ModelLoader.load_model()
        
        # Setup UI controls
        search_params = UI.setup_sidebar(df)
        
        if st.sidebar.button("Get Recommendations"):
            with st.spinner("Finding recommendations..."):
                try:
                    original_title, recommendations = Recommender.recommend_manga(
                        title=search_params['title_query'],
                        df=df,
                        model=model,
                        index=index,
                        k=search_params['k'],
                        genre_filter=search_params['genre_filter'],
                        type_filter=search_params['type_filter'],
                        allow_low_popularity=search_params['allow_low_popularity'],
                        sort_by=search_params['sort_by']
                    )
                    
                    if recommendations is None:
                        st.error(f"No recommendations found for: {search_params['title_query']}")
                    else:
                        st.success(f"Found {len(recommendations)} recommendations for: **{original_title}**")
                        
                        # Get AniList data for main title
                        main_data = APIHandler.get_anilist_data(original_title)
                        
                        # Display main title info
                        if main_data:
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                ImageHandler.safe_image_display(
                                    main_data.get('coverImage', {}).get('large'),
                                    300, 
                                    original_title
                                )
                            with col2:
                                st.markdown(f"### {original_title}")
                                if main_data.get('meanScore'):
                                    st.markdown(f"**Rating:** ⭐ {main_data['meanScore']}/100")
                                if main_data.get('siteUrl'):
                                    st.markdown(f"**[View on AniList]({main_data['siteUrl']})**")
                        
                        st.divider()
                        
                        # Display recommendations
                        progress_bar = st.progress(0)
                        for i, rec in enumerate(recommendations, 1):
                            try:
                                # Get AniList data with progress update
                                progress_bar.progress(i/len(recommendations))
                                anilist_data = APIHandler.get_anilist_data(rec['title'])
                                UI.display_recommendation(rec, anilist_data)
                            except Exception as e:
                                logger.error(f"Failed to display recommendation {i}: {str(e)}")
                                continue
                        progress_bar.empty()
                        
                except Exception as e:
                    logger.error(f"Recommendation process failed: {str(e)}")
                    st.error("Failed to generate recommendations. Please try again.")
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("A critical error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main()
