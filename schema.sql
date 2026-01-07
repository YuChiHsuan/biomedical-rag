-- schema.sql
-- Database schema for the Biomedical RAG System
-- Description: Stores articles and query history for caching and analytics

-- --------------------------------------------------------
-- Database Creation (Optional - uncomment if needed)
-- --------------------------------------------------------
-- CREATE DATABASE IF NOT EXISTS biomedical_rag
-- CHARACTER SET utf8mb4
-- COLLATE utf8mb4_unicode_ci;
-- 
-- USE biomedical_rag;

-- --------------------------------------------------------
-- 1. ARTICLES TABLE
-- Stores articles/documents fetched from PubMed (or other sources in future...)
-- --------------------------------------------------------

CREATE TABLE IF NOT EXISTS articles (
    article_id VARCHAR(50) PRIMARY KEY COMMENT 'Unique article identifier (e.g. pubmed_12345)',
    title TEXT COMMENT 'Article title',
    abstract TEXT COMMENT 'Article abstract',
    article_embedding JSON COMMENT 'Embedding vector as JSON array',
    source VARCHAR(50) COMMENT 'Source (PubMed, wiki, etc.)',
    retrieved_at DATETIME COMMENT 'Timestamp when article was retrieved',
    
    INDEX idx_retrieved_at (retrieved_at),
    INDEX idx_source (source(50)),
    INDEX idx_title (title(255)),
    INDEX idx_abstract (abstract(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Biomedical articles with embeddings';


-- --------------------------------------------------------
-- 2. QUERIES TABLE
-- Stores historical queries and their generated answers for caching
-- --------------------------------------------------------

CREATE TABLE IF NOT EXISTS queries (
    query_id CHAR(36) PRIMARY KEY COMMENT 'UUID for query',
    original_user_input TEXT COMMENT 'Original user question (in any language)',
    user_input TEXT COMMENT 'Translated user question (in English)',
    extracted_keywords TEXT COMMENT 'JSON of extracted keywords',
    search_method VARCHAR(50) COMMENT 'Search method used (e.g. db, pubmed, db+pubmed)',
    top_articles TEXT COMMENT 'JSON array of used article IDs',
    generated_answer TEXT COMMENT 'Generated answer (in Traditional Chinese)',
    answer_en TEXT COMMENT 'Translated generated answer (in English)',
    answer_embedding JSON COMMENT 'Answer embedding as JSON array',
    created_at DATETIME COMMENT 'Query timestamp',
    query_embedding JSON COMMENT 'Query embedding as JSON array',
    confidence_score FLOAT COMMENT 'RAG confidence (avg article similarity)',
    answer_relevance FLOAT COMMENT 'Answer-query relevance score',
    response_time FLOAT COMMENT 'Total response time in seconds',
    
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='User queries and cached answers';


-- --------------------------------------------------------
-- 3. OPTIONAL: Analytics View
-- Provides aggregated statistics for monitoring
-- --------------------------------------------------------

CREATE OR REPLACE VIEW query_analytics AS
SELECT 
    DATE(created_at) as query_date,
    search_method,
    COUNT(*) as query_count,
    AVG(confidence_score) as avg_confidence,
    AVG(answer_relevance) as avg_relevance,
    AVG(response_time) as avg_response_time,
    MIN(response_time) as min_response_time,
    MAX(response_time) as max_response_time
FROM queries
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY DATE(created_at), search_method
ORDER BY query_date DESC, search_method;
