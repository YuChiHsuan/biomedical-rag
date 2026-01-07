"""
Biomedical RAG System
A retrieval-augmented generation system for biomedical question answering.

Features:
- Local fine-tuned embedding model for domain-specific semantic search
- Binary classifier for security gate (biomedical vs non-biomedical)
- Hybrid search strategy (local database + PubMed API)
- Query caching to reduce API calls and improve response time
- Multi-language support (auto-translation)
"""

import os
import time
import json
import uuid
import re
import requests
import datetime
import random
import hashlib
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
from bs4 import BeautifulSoup
from langdetect import detect
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import joblib

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI package is required. Install with: pip install openai")

# ==================== Configuration ====================
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "127.0.0.1"),
    "port": int(os.environ.get("DB_PORT", 3306)),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "<db_password>"),
    "database": os.environ.get("DB_NAME", "<db_name>"),
    "charset": "utf8mb4"
}

# Classifier threshold for security gate (higher = stricter)
CLASSIFIER_THRESHOLD = float(os.environ.get("CLASSIFIER_THRESHOLD", 0.70))

# ==================== Model Loading ====================

# Load fine-tuned embedding model
try:
    EMB_MODEL = SentenceTransformer("./biomedical-rag-finetuned")
    EMB_DIM = EMB_MODEL.get_sentence_embedding_dimension()
    print(f"[Embedding] Loaded fine-tuned model, dimension: {EMB_DIM}")
except Exception as e:
    print(f"[Embedding] Fine-tuned model not found. Using base BioBERT: {e}")
    EMB_MODEL = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    EMB_DIM = EMB_MODEL.get_sentence_embedding_dimension()

# Load binary classifier for security gate
try:
    LOCAL_CLASSIFIER = joblib.load("binary_classifier_finetuned.pkl")
    print("[Classifier] Loaded local binary classifier")
    print(f"[Classifier] Security threshold: {CLASSIFIER_THRESHOLD}")
except Exception as e:
    print(f"[Classifier] Failed to load classifier: {e}")
    LOCAL_CLASSIFIER = None

# ==================== Utility Functions ====================

def _clean_text(text: str, max_chars: int = 3000) -> str:
    """
    Normalize and truncate text for processing.
    
    Args:
        text: Input text string
        max_chars: Maximum characters to keep
        
    Returns:
        Cleaned text string
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().replace("\x00", " ")
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def get_embedding(text: str) -> np.ndarray:
    """
    Generate embedding vector using local fine-tuned model.
    
    Args:
        text: Input text to embed
        
    Returns:
        Numpy array of embedding vector
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return np.zeros(EMB_DIM, dtype=float)

    if EMB_MODEL is not None:
        try:
            emb = EMB_MODEL.encode(
                cleaned,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return emb
        except Exception as e:
            print(f"[Embedding] Error: {e}")
            return np.zeros(EMB_DIM, dtype=float)
    else:
        return np.zeros(EMB_DIM, dtype=float)


def translate_to_en(text: str, lang: str = "zh-TW") -> str:
    """
    Translate text to English using Google Translator.
    
    Args:
        text: Text to translate
        lang: Language of text (assumed Traditional Chinese)
        
    Returns:
        Translated English text
    """
    try:
        return GoogleTranslator(source=lang, target="en").translate(text)
    except Exception as e:
        print(f"[Translation] Error: {e}")
        return text


def chunk_text(text: str, max_len: int = 1000) -> List[str]:
    """
    Split long text into chunks for messaging platforms (LINE) if needed.
    
    Args:
        text: Text to split
        max_len: Maximum length per chunk
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= max_len:
        return [text] if text else []
    
    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        chunk = text[start:end]
        
        # Try to break at newline if possible
        if end < len(text):
            nl_pos = chunk.rfind('\n')
            if nl_pos > (max_len * 2 // 3):
                parts.append(text[start:start + nl_pos + 1])
                start = start + nl_pos + 1
                continue
        
        parts.append(chunk)
        start = end
    return parts


# ==================== Keyword Extractor ====================

class KeywordExtractor:
    """
    Extract biomedical keywords using GPT with disk-based caching.
    """

    def __init__(self, cache_path: str = "keywords_cache.json"):
        """
        Initialize keyword extractor with cache.
        
        Args:
            cache_path: Path to JSON cache file
        """
        self.cache_path = cache_path
        self.keywords_cache: Dict[str, Any] = {}
        self.load_cache()

    def load_cache(self):
        """Load keyword cache from disk."""
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                self.keywords_cache = json.load(f)
            print(f"[Cache] Loaded {len(self.keywords_cache)} cached keywords")
        except FileNotFoundError:
            self.keywords_cache = {}
    
    def save_cache(self):
        """Save keyword cache to disk."""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.keywords_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Cache] Save error: {e}")
    
    def extract_keywords(self, text: str) -> Dict[str, List[str]]:
        """
        Extract keywords, MeSH terms, and title/abstract terms using GPT.
        Results are cached to reduce API calls.
        
        Args:
            text: Query text to extract keywords from
            
        Returns:
            Dictionary with keys: keywords, mesh_terms, tiab_terms
        """
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
        if cache_key in self.keywords_cache:
            return self.keywords_cache[cache_key]

        try:
            prompt = f"""Extract biomedical keywords and terms for PubMed search from this text: {text}

Return ONLY a JSON object with exactly this format (no extra text):
{{
  "keywords": ["actual_keyword1", "actual_keyword2", "actual_keyword3"],
  "mesh_terms": ["MeSH Term 1", "MeSH Term 2"],
  "tiab_terms": ["term for title/abstract search"]
}}

Important: 
- Do NOT include the words "keywords", "mesh_terms", or "tiab_terms" as actual content
- Focus on biomedical/scientific terms only
- Limit to 1-3 relevant items per category
- Use proper medical terminology
"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a biomedical search expert. Extract relevant keywords and MeSH terms for PubMed literature search. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback: extract quoted strings
                kws = re.findall(r'"([^"]+)"', result_text)
                kws = [k for k in kws if k.lower() not in ['keywords', 'mesh_terms', 'tiab_terms']]
                result = {
                    "keywords": kws[:3],
                    "mesh_terms": kws[:3],
                    "tiab_terms": kws[:3]
                }

            # Clean results to remove any leaked JSON keys
            def clean_terms(terms_list):
                cleaned = []
                for term in terms_list:
                    term_lower = term.lower().strip()
                    if term_lower not in ['keywords', 'mesh_terms', 'tiab_terms', 'keyword', 'mesh_term', 'tiab_term', ''] and len(term.strip()) > 1:
                        cleaned.append(term.strip())
                return cleaned[:3]

            result = {
                "keywords": clean_terms(result.get("keywords", [])),
                "mesh_terms": clean_terms(result.get("mesh_terms", [])),
                "tiab_terms": clean_terms(result.get("tiab_terms", []))
            }

            # Cache result
            self.keywords_cache[cache_key] = result
            self.save_cache()

            return result

        except Exception as e:
            print(f"[Keywords] Error: {e}")
            return {"keywords": [], "mesh_terms": [], "tiab_terms": []}


# ==================== Main RAG System ====================

class BiomedicalRAGBot:
    """
    Biomedical RAG system with hybrid search and security gate.
    """

    def __init__(self, db_config: dict = DB_CONFIG):
        """
        Initialize RAG bot with database connection and models.
        
        Args:
            db_config: Database configuration dictionary
        """
        self.conn = self._init_database(db_config)
        self.keyword_extractor = KeywordExtractor()
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # Query cache thresholds
        self.query_cache_q_threshold = 0.85  # Query similarity threshold
        self.query_cache_a_threshold = 0.65  # Answer relevance threshold
        self.query_cache_min_count = 2       # Minimum matches needed
        
        # Search parameters
        self.db_article_sim_threshold = 0.30  # Minimum similarity for DB articles
        self.db_article_limit = 20            # Max articles to fetch from DB
        self.db_article_count = 3             # Preferred article count from DB
        self.pubmed_retmax = 5                # Max PubMed articles to fetch

    def _init_database(self, db_config: dict):
        """
        Initialize MySQL database connection and ensure tables exist.
        Tables are created if they don't exist (see schema.sql for manual setup).

        Args:
            db_config: Database configuration
            
        Returns:
            pymysql connection object
        """
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()

        # Note: Run schema.sql manually for better control
        # Create articles table if not exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            article_id VARCHAR(50) PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            article_embedding JSON,
            source VARCHAR(50),
            retrieved_at DATETIME,
            INDEX idx_retrieved_at (retrieved_at),
            INDEX idx_source (source(50)),
            INDEX idx_title (title(255)),
            INDEX idx_abstract (abstract(255))
        )
        """)
        
        # Create queries table if not exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            query_id CHAR(36) PRIMARY KEY,
            original_user_input TEXT,
            user_input TEXT,
            extracted_keywords TEXT,
            search_method VARCHAR(50),
            top_articles TEXT,
            generated_answer TEXT,
            answer_en TEXT,
            answer_embedding JSON,
            created_at DATETIME,
            query_embedding JSON,
            confidence_score FLOAT,
            answer_relevance FLOAT,
            response_time FLOAT,
            INDEX idx_created_at (created_at)
        )
        """)
            
        conn.commit()
        cursor.close()
        print("[Database] Tables initialized successfully")
        return conn

    def get_cached_embedding(self, text: str, cache_key: Optional[str] = None) -> np.ndarray:
        """
        Get embedding with memory caching to avoid redundant computations.
        
        Args:
            text: Text to embed
            cache_key: Optional cache key (auto-generated if None)
            
        Returns:
            Embedding vector
        """
        if cache_key is None:
            cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
        if cache_key not in self._embedding_cache:
            self._embedding_cache[cache_key] = get_embedding(text)
        return self._embedding_cache[cache_key]
    
    def _check_query_cache(self, q_emb: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Check if a similar query has been answered before.
        
        Args:
            q_emb: Query embedding vector
            
        Returns:
            Cached answer if found, None otherwise
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT user_input, generated_answer, query_embedding, answer_embedding
                FROM queries 
                WHERE query_embedding IS NOT NULL 
                AND answer_embedding IS NOT NULL
                AND JSON_LENGTH(query_embedding) > 0
                AND JSON_LENGTH(answer_embedding) > 0
                AND created_at >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
                ORDER BY created_at DESC 
            """)
            rows = cursor.fetchall()
            cursor.close()
            
            if not rows:
                return None, None
                
            similar_answers = []
            for prev_q, prev_ans, prev_q_emb_json, prev_a_emb_json in rows:
                if not prev_q_emb_json or not prev_a_emb_json:
                    continue
                try:
                    prev_q_emb = np.array(json.loads(prev_q_emb_json))
                    prev_a_emb = np.array(json.loads(prev_a_emb_json))
                    
                    q_sim = cosine_similarity(q_emb.reshape(1, -1), prev_q_emb.reshape(1, -1))[0][0]
                    if q_sim >= self.query_cache_q_threshold:
                        a_sim = cosine_similarity(q_emb.reshape(1, -1), prev_a_emb.reshape(1, -1))[0][0]
                        if a_sim >= self.query_cache_a_threshold:
                            similar_answers.append(prev_ans)
                            if len(similar_answers) >= self.query_cache_min_count:
                                print(f"[Cache Hit] q_sim={q_sim:.3f}, a_sim={a_sim:.3f}")
                                return random.choice(similar_answers), a_sim
                except Exception as e:
                    print(f"[Cache] Error processing row: {e}")
                    continue
            return None, None
        except Exception as e:
            print(f"[Cache] Error: {e}")
            return None, None

    def _search_database(self, keywords: Dict[str, List[str]], q_emb: np.ndarray, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search local database using keyword matching and embedding similarity.
        
        Args:
            keywords: Extracted keywords dict
            q_emb: Query embedding
            limit: Maximum articles to fetch
        
        Returns:
            List of relevant articles with similarity scores
        """
        cursor = self.conn.cursor()
        
        # Collect keywords and tiab_terms
        all_keywords = []
        for key in ['keywords', 'tiab_terms']:
            all_keywords.extend(keywords.get(key, []))
        all_keywords = list(set(all_keywords))
        
        if not all_keywords:
            cursor.close()
            return []
            
        # Build search query
        search_terms = []
        params = []
        for kw in all_keywords:
            kw_clean = kw.strip()
            search_terms.append("(title LIKE %s OR abstract LIKE %s)")
            params.extend([f"%{kw_clean}%", f"%{kw_clean}%"])
            
        sql = f"""
            SELECT article_id, title, abstract, article_embedding, source, retrieved_at
            FROM articles
            WHERE ({' OR '.join(search_terms)})
            AND article_embedding IS NOT NULL
            AND JSON_LENGTH(article_embedding) > 0
            AND retrieved_at >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
            ORDER BY retrieved_at DESC
            LIMIT %s
        """
        params.append(limit)
        
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        except Exception as e:
            print(f"[DB Search] Error: {e}")
            cursor.close()
            return []
        cursor.close()

        # Calculate similarities
        articles = []
        for r in rows:
            try:
                emb = np.array(json.loads(r[3]))
                sim = cosine_similarity(q_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
                
                if sim >= self.db_article_sim_threshold:
                    articles.append({
                        "article_id": r[0],
                        "title": r[1] or "",
                        "abstract": r[2] or "",
                        "article_embedding": emb,
                        "source": r[4] or "DB",
                        "retrieved_at": r[5].isoformat() if isinstance(r[5], datetime.datetime) else str(r[5]),
                        "similarity": float(sim)
                    })
            except Exception as e:
                print(f"[DB Search] Error processing row: {e}")
                continue
                
        articles.sort(key=lambda x: x["similarity"], reverse=True)
        articles = articles[:self.db_article_count]
        
        print(f"[DB Search] Found {len(articles)}/{len(rows)} articles (threshold: {self.db_article_sim_threshold})")
        return articles

    def _search_pubmed(self, keywords: Dict[str, List[str]], q_emb: np.ndarray, retmax: int = 5, exclude_articles: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Search PubMed using NCBI E-utilities API (with deduplication).
        
        Args:
            keywords: Extracted keywords dict
            q_emb: Query embedding for relevance scoring
            retmax: Maximum results to fetch
            exclude_articles: Articles to exclude (already in DB articles)
            
        Returns:
            List of PubMed articles with similarity scores
        """
        if exclude_articles is None:
            exclude_articles = []
        
        keyw_terms = keywords.get("keywords", [])[:3]
        mesh_terms = keywords.get("mesh_terms", [])[:3]
        tiab_terms = keywords.get("tiab_terms", [])[:3]
        
        query_parts = []
        query_parts.extend([f'"{term}"' for term in keyw_terms])
        query_parts.extend([f'"{term}"[MeSH Terms]' for term in mesh_terms])
        query_parts.extend([f'"{term}"[tiab]' for term in tiab_terms])
        
        if not query_parts:
            return []
            
        term = " OR ".join(query_parts)
        esearch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=pubmed&term={term}&reldate=1095&datetype=pdat&retmode=json&retmax={retmax}"
        )
        
        try:
            r = requests.get(esearch_url, timeout=8)
            r.raise_for_status()
            data = r.json()
            pmids = data["esearchresult"]["idlist"]
        except Exception as e:
            print(f"[PubMed Search] Error: {e}")
            return []

        if not pmids:
            return []
        
        # Deduplicate: filter out PMIDs that are already in exclude_articles
        existing_pmids = {art['article_id'].replace('pubmed_', '') for art in exclude_articles if 'pubmed_' in art.get('article_id', '')}
        new_pmids = [p for p in pmids if p not in existing_pmids]

        if not new_pmids:
            print(f"[PubMed Search] All {len(pmids)} articles already in database, skipping fetch")
            return []
        
        if len(existing_pmids) > 0:
            print(f"[PubMed Search] Filtered {len(pmids)-len(new_pmids)} duplicate articles, fetching {len(new_pmids)} new articles")

        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=pubmed&id={','.join(new_pmids)}&retmode=xml"
        )
        try:
            text = requests.get(fetch_url, timeout=15).text
            return self._parse_pubmed_xml(text, new_pmids, q_emb)
        except Exception as e:
            print(f"[PubMed Fetch] Error: {e}")
            return []

    def _parse_pubmed_xml(self, xml_content: str, pmids: List[str], q_emb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Parse PubMed XML and generate embeddings in parallel.
        
        Args:
            xml_content: Raw XML from PubMed
            pmids: List of PubMed IDs
            q_emb: Query embedding for similarity calculation
            
        Returns:
            List of parsed articles with embeddings
        """
        parsed_results = []
        for i, article_xml in enumerate(xml_content.split("<PubmedArticle>")[1:]):
            try:
                title_match = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', article_xml, re.DOTALL)
                if not title_match:
                    continue
                title = BeautifulSoup(title_match.group(1), "html.parser").get_text(" ", strip=True)[:200]
                
                abstract_match = re.search(r'<AbstractText[^>]*>(.*?)</AbstractText>', article_xml, re.DOTALL)
                if not abstract_match:
                    continue
                abstract = BeautifulSoup(abstract_match.group(1), "html.parser").get_text(" ", strip=True)[:3000]
                
                pmid = pmids[i]
                parsed_results.append((pmid, title, abstract))
            except Exception:
                continue
        
        def _embed_article(item):
            """Helper function for parallel embedding generation."""
            pmid, title, abstract = item
            emb = get_embedding(title + " " + abstract)
            sim = cosine_similarity(q_emb.reshape(1, -1), emb.reshape(1, -1))[0][0] if emb is not None else 0.0

            return {
                "article_id": f"pubmed_{pmid}",
                "title": title,
                "abstract": abstract,
                "article_embedding": emb,
                "source": "PubMed",
                "retrieved_at": datetime.datetime.now().isoformat(),
                "pmid": pmid,
                "similarity": float(sim)
            }

        articles = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_embed_article, item) for item in parsed_results]
            for f in futures:
                try:
                    articles.append(f.result())
                except Exception:
                    continue
                    
        print(f'[PubMed Search] Found {len(articles)} articles')
        return articles

    def search_strategy(self, keywords: Dict[str, List[str]], q_emb: np.ndarray) -> Tuple[List[Dict[str, Any]], str]:
        """
        Hybrid search strategy: Database first, supplement with PubMed if needed.
        
        Args:
            keywords: Extracted keywords
            q_emb: Query embedding
            
        Returns:
            Tuple of (articles list sorted by similarity scores, search method string)
        """
        search_methods = []
        all_articles = []

        # Search local database first
        db_start = time.time()
        db_articles = self._search_database(keywords, q_emb, limit=self.db_article_limit)
        db_time = time.time() - db_start

        if db_articles:
            all_articles.extend(db_articles)
            search_methods.append("database")
            print(f"[DB Search] Time: {db_time:.2f}s, {len(db_articles)} articles")
        
        # Supplement with PubMed if insufficient
        if len(db_articles) < 3:
            pubmed_start = time.time()
            pubmed_articles = self._search_pubmed(keywords, q_emb, retmax=self.pubmed_retmax, exclude_articles=db_articles)
            pubmed_time = time.time() - pubmed_start

            if pubmed_articles:
                print(f'[PubMed Search] Time: {pubmed_time:.2f}s, saving {len(pubmed_articles)} articles')
                for article in pubmed_articles:
                    self._save_article(article)
                
                all_articles.extend(pubmed_articles)
                search_methods.append("pubmed")
        
        top_articles = sorted(all_articles, key=lambda x: x.get("similarity", 0), reverse=True)
        return top_articles, "+".join(search_methods) if search_methods else "none"

    def _save_article(self, article: Dict[str, Any]):
        """
        Save article to database (uses REPLACE to avoid duplicates).
        
        Args:
            article: Article dictionary with embeddings
        """
        try:
            cursor = self.conn.cursor()
            emb_json = json.dumps(article["article_embedding"].tolist()) if isinstance(article["article_embedding"], np.ndarray) else None
            
            cursor.execute("""
                REPLACE INTO articles
                (article_id, title, abstract, article_embedding, source, retrieved_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                str(article["article_id"]),
                article.get("title", ""),
                article.get("abstract", ""),
                emb_json,
                article.get("source", ""),
                datetime.datetime.now()
            ))
            self.conn.commit()
            cursor.close()
        except Exception as e:
            print(f"[Save Article] Error: {e}")
            try:
                self.conn.rollback()
            except:
                pass

    def _choose_num_articles(self, user_question: str, top_articles: List[Dict[str, Any]], keywords: Dict[str, List[str]]) -> int:
        """
        Choose number of articles to use (1-3) based on query complexity.
        
        Args:
            user_question: Original user query
            top_articles: Retrieved articles sorted by similarity
            keywords: Extracted keywords
        
        Returns:
            Number of articles to use (1-3)
        """
        if not top_articles:
            return 0
            
        sims = [a.get("similarity", 0.0) for a in top_articles[:3]]
        top_sim = sims[0] if sims else 0.0
        q_len = len(user_question.split())
        
        # Check for complex biomedical terms
        complex_terms = [
            "mechanism", "pathogenesis", "treatment", "efficacy", 
            "compare", "meta-analysis", "risk", "protocol", "optimization", "adverse"
        ]
        q_lower = user_question.lower()
        kw_flat = " ".join([k.lower() for lst in keywords.values() for k in lst]) if keywords else ""
        contains_complex = any(term in q_lower or term in kw_flat for term in complex_terms)
        
        # Decision logic
        if contains_complex:
            chosen = 3
        elif (top_sim >= 0.7) or (q_len <= 10 and top_sim >= 0.4):
            chosen = 1
        else:
            chosen = 2
            
        print(f'[Article Selection] Using {chosen} articles (top_sim={top_sim:.3f})')
        return chosen

    def _generate_answer(self, user_question: str, top_articles: List[Dict[str, Any]], n_use: int) -> str:
        """
        Generate answer using GPT with retrieved articles as context.
        
        Args:
            user_question: Original user question
            top_articles: Retrieved articles
            n_use: Number of articles to use
        
        Returns:
            Generated answer in Traditional Chinese with disclaimer
        """
        if not top_articles:
            return ("很抱歉，小幫手無法找到足夠文獻來回答您的問題。"
                   "建議您嘗試別的問法，或是諮詢專業人員。\n\n此回答僅供參考，不構成醫療建議。")
                   
        context_parts = []
        for i, article in enumerate(top_articles[:n_use], 1):
            context_parts.append(f"Article{i}: {article['title']}\n{article['abstract']}")
        context_text = "\n\n".join(context_parts)

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": """You are a professional biomedical researcher.
Reply in 繁體中文, concise and clear (<=350 words).

Rules:
- Fix user typos automatically.
- Always use Taiwanese vocabularies: (品質, 訊息, 活化, 管道, 標靶, etc.) instead of Chinese vocabularies (質量, 信息, 激活, 渠道, 靶標, etc.).
- Users do NOT know the content of the reference articles. Do NOT say "根據文章一/二" or similar; instead, integrate the knowledge directly.
- If the user explicitly asks for 資料來源, 來源連結, 文章, reference, source, citation, link: reply first with: "很抱歉，小幫手目前無法提供資料來源。" and then continue to answer their question.
- Keep the tone professional but easy to understand."""},
                    {"role": "user", "content": f"問題: {user_question}\n\n參考內容:\n{context_text}"}
                ],
                max_tokens=600
            )
            answer = resp.choices[0].message.content.strip()
            disclaimer = "\n\n本回答僅供參考，不構成醫療建議。如有疑慮，請諮詢專業人員。"
            return answer + disclaimer
        except Exception as e:
            print(f"[Answer Generation] Error: {e}")
            return "發生錯誤，請稍後再試。\n\n本回答僅供參考，不構成醫療建議。"

    def _is_biomedical_question(self, q_emb: np.ndarray) -> Tuple[bool, float]:
        """
        Use local binary classifier to determine if query is biomedical.
        Acts as a security gate to prevent off-topic queries.
        
        Args:
            q_emb: Query embedding vector
        
        Returns:
            Tuple of (is_biomedical: bool, confidence: float)
        """
        if LOCAL_CLASSIFIER is None or EMB_MODEL is None:
            print("[Security Gate] Local models not loaded, rejecting query")
            return False, 0.0
            
        try:            
            q_emb_reshaped = q_emb.reshape(1, -1)
            proba = LOCAL_CLASSIFIER.predict_proba(q_emb_reshaped)[0, 1]
            is_biomed = (proba >= CLASSIFIER_THRESHOLD)
            
            print(f"[Security Gate] Probability={proba:.4f}, Threshold={CLASSIFIER_THRESHOLD:.4f}, Result={is_biomed}")
            
            return is_biomed, float(proba)
        
        except Exception as e:
            print(f"[Security Gate] Error: {e}")
            return False, 0.0

    def _is_other_question(self, question: str) -> str:
        """
        Use GPT to classify non-biomedical queries for rejection.
        Returns category code: 0=greeting, 1=who are you, 2=company/product, 3=diagnosis, 4=other
        
        Args:
            question: User query text
            
        Returns:
            Category code as string ('0'-'4')
        """
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": """You are a strict classifier. Reply ONLY with 0, 1, 2, 3, or 4.

0 = Greetings.
1 = Asking who you are.
2 = Questions about specific companies, products, or pipelines.
3 = Personal medical prediction, diagnosis, or individualized medical advice.
4 = Everything else (weather, stock, physics, chemistry, astronomy, math, finance, meaningless text)

Examples:
User: "hello" → 0
User: "who are you?" → 1
User: "What is Company A doing?" → 2
User: "Will I have cancer?" → 3
User: "Stock price of Tesla today?" → 4
User: "Explain Newton's law" → 4
"""},
                    {"role": "user", "content": question}
                ],
                max_tokens=1,
                temperature=0
            )
            result = resp.choices[0].message.content.strip()
            if result not in "01234":
                return '4'
            return result[0]
        except Exception as e:
            print(f"[Question Classification] Error: {e}")
            return '4'

    def _save_query(self, original_user_input: str, user_input: str, query_emb: np.ndarray, 
                   keywords: Dict, search_method: str, top_articles: List[Dict[str, Any]], 
                   answer: str, answer_en: str, answer_emb: np.ndarray, 
                   confidence: float, answer_relevance: float, response_time: float):
        """
        Save query and response to database for caching and analytics.
        
        Args:
            original_user_input: Original query (any language)
            user_input: Translated English query
            query_emb: Query embedding
            keywords: Extracted keywords
            search_method: Search method used (db/pubmed/db+pubmed)
            top_articles: Retrieved article IDs
            answer: Generated answer (reply to user)
            answer_en: English translation of answer (for embedding)
            answer_emb: Answer embedding
            confidence: RAG confidence score (average similarity of selected articles)
            answer_relevance: Answer-query relevance (similarity between answer and query)
            response_time: Total response time
        """
        try:
            cursor = self.conn.cursor()
            qid = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO queries 
                (query_id, original_user_input, user_input, query_embedding, extracted_keywords, search_method, 
                 top_articles, generated_answer, answer_en, answer_embedding, confidence_score, answer_relevance, response_time, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                qid, 
                original_user_input, 
                user_input, 
                json.dumps(query_emb.tolist()), 
                json.dumps(keywords), 
                search_method,
                json.dumps([a["article_id"] for a in top_articles]), 
                answer, 
                answer_en, 
                json.dumps(answer_emb.tolist()),
                confidence, 
                answer_relevance, 
                response_time, 
                datetime.datetime.now()
            ))
            self.conn.commit()
            cursor.close()
            print(f'[Query Saved] ID: {qid}, method: {search_method}')
        except Exception as e:
            print(f"[Save Query] Error: {e}")
            try:
                self.conn.rollback()
            except:
                pass

    def answer_question(self, user_question: str) -> Tuple[str, List[Dict[str, Any]], float, float, float]:
        """
        Main pipeline: translation → security gate → cache check → search → generate → save.
        
        Args:
            user_question: User's question in any language
            
        Returns:
            Tuple of (answer, articles, confidence, answer_relevance, response_time)
        """
        total_start = time.time()
        org_question = user_question

        # Translate to English if needed
        try:
            lang = detect(org_question)
            if lang != 'en':
                user_question = translate_to_en(user_question)
                print(f"[Translation] {lang} → en: {user_question}")
        except Exception as e:
            print(f"[Language Detection] Error: {e}")

        # Security Gate: Check if biomedical question
        q_emb = self.get_cached_embedding(user_question, f"q_{hashlib.md5(user_question.encode()).hexdigest()[:16]}")
        is_biomed, confidence_raw = self._is_biomedical_question(q_emb)

        if not is_biomed:
            # Classify non-biomedical query for friendly response
            try:
                cls_code = self._is_other_question(user_question)
            except Exception as e:
                print(f"[Classification] Error: {e}")
                cls_code = '4'

            # Return appropriate rejection message
            if cls_code == '0':
                return "您好！請輸入生醫相關問題，小幫手會協助搜尋與回答～", [], 0.0, 0.0, time.time() - total_start
            elif cls_code == '1':
                return "我是生醫QA小幫手，可以幫忙回答生醫相關問題。（答案僅供參考）", [], 0.0, 0.0, time.time() - total_start
            elif cls_code == '2':
                return "很抱歉，小幫手目前無法提供特定公司或產品的資訊！", [], 0.0, 0.0, time.time() - total_start
            elif cls_code == '3':
                return "很抱歉，小幫手無法提供個人化的醫療診斷或預測，建議您尋求專業醫師的協助。", [], 0.0, 0.0, time.time() - total_start
            else:
                return "很抱歉，這看起來不是生醫相關問題喔！小幫手無法提供相關查詢～", [], 0.0, 0.0, time.time() - total_start
        
        # ===== Passed Security Gate =====
        
        # Check cache for similar queries
        cached_answer, cached_answer_relevance = self._check_query_cache(q_emb)
        if cached_answer and cached_answer_relevance:
            print("[Cache Hit] Returning cached answer")
            return cached_answer, [], 1.0, cached_answer_relevance, time.time() - total_start

        # Extract keywords
        keywords = self.keyword_extractor.extract_keywords(user_question)
        print(f"[Keywords] {keywords}")

        # Hybrid search strategy
        top_articles, search_method = self.search_strategy(keywords, q_emb)

        # Select number of articles to use
        n_use = self._choose_num_articles(user_question, top_articles, keywords)
        top_selected = top_articles[:n_use]
        
        # Generate answer
        answer = self._generate_answer(org_question, top_selected, n_use)
        
        # Calculate metrics
        answer_en = translate_to_en(answer)
        answer_emb = get_embedding(answer_en)
        
        # RAG confidence: average similarity of selected articles
        confidence = float(np.mean([a.get("similarity", 0) for a in top_selected])) if top_selected else 0.0
        
        # Answer relevance: similarity between query and answer
        answer_relevance = float(cosine_similarity([q_emb], [answer_emb])[0][0]) if answer_emb.shape[0] == q_emb.shape[0] else 0.0

        total_time = time.time() - total_start

        # Save query for caching and analytics
        self._save_query(org_question, user_question, q_emb, keywords, search_method, 
                       top_selected, answer, answer_en, answer_emb, 
                       confidence, answer_relevance, total_time)
        
        print(f"[Pipeline] Completed in {total_time:.1f}s, confidence={confidence:.3f}, relevance={answer_relevance:.3f}")
        return answer, top_articles, confidence, answer_relevance, total_time

    def close(self):
        """Clean up database connection and clear caches."""
        try:
            if self.conn:
                self.conn.close()
                print("[Database] Connection closed")
        except Exception:
            pass
        self._embedding_cache.clear()


# ==================== Demo & Testing ====================

def main_demo():
    """
    Command-line demo for testing the RAG system.
    """
    print("=" * 60)
    print("Biomedical RAG Bot - Demo Mode")
    print("=" * 60)
    
    bot = BiomedicalRAGBot()
    
    try:
        test_questions = [
            "你好",
            "什麼是薛丁格的貓",
            "影響表觀遺傳學的蛋白質有哪些", 
            "新冠肺炎疫苗有什麼副作用"
        ]
        
        for idx, q in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"Test {idx}/{len(test_questions)}")
            print(f"Q: {q}")
            print(f"{'-'*60}")
            
            a, arts, conf, rele, t = bot.answer_question(q)
            
            print(f"\nA: {a[:300]}{'...' if len(a) > 300 else ''}")
            print(f"\n[Stats]")
            print(f"  - Articles found: {len(arts)}")
            print(f"  - Confidence: {conf:.3f}")
            print(f"  - Relevance: {rele:.3f}")
            print(f"  - Response time: {t:.2f}s")
            
            if arts:
                print(f"\n[Top Article]")
                top = arts[0]
                print(f"  - ID: {top.get('article_id', 'N/A')}")
                print(f"  - Similarity: {top.get('similarity', 0):.3f}")
                print(f"  - Source: {top.get('source', 'N/A')}")
            
            time.sleep(1)  # Avoid API rate limit
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError in demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bot.close()
        print("\n" + "="*60)
        print("Demo completed")
        print("="*60)


if __name__ == "__main__":
    main_demo()