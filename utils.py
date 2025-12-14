"""
Complete Utility Functions for LLM Evaluation Pipeline
Includes: embeddings, similarity, claim extraction, entity matching, grounding
"""

import os
import re
import numpy as np
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
import tiktoken
from dotenv import load_dotenv

load_dotenv()

# Initialize embedding model (cached globally)
_embedding_model = None

def get_embedding_model():
    """Get or initialize the embedding model (singleton pattern)"""
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv('EMBEDDING_MODEL') or 'all-MiniLM-L6-v2'
        print(f"Loading embedding model: {model_name}...")
        _embedding_model = SentenceTransformer(model_name)
        print("✅ Embedding model loaded!")
    return _embedding_model


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts"""
    if not texts:
        return np.array([])
    
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return np.array([])
    
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    if vec1 is None or vec2 is None:
        return 0.0
    
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    
    if vec1_norm == 0 or vec2_norm == 0:
        return 0.0
    
    vec1_normalized = vec1 / vec1_norm
    vec2_normalized = vec2 / vec2_norm
    
    similarity = np.dot(vec1_normalized, vec2_normalized)
    return float(max(0.0, min(1.0, similarity)))


def cosine_similarity_batch(query_embedding: np.ndarray, 
                            doc_embeddings: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between one query and multiple documents"""
    if len(doc_embeddings) == 0:
        return np.array([])
    
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    similarities = np.dot(doc_norms, query_norm)
    similarities = np.clip(similarities, 0.0, 1.0)
    
    return similarities


# ==================== SENTENCE & CLAIM EXTRACTION ====================

def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text with improved splitting"""
    # Handle common abbreviations
    text = text.replace('Dr.', 'Dr~')
    text = text.replace('Mr.', 'Mr~')
    text = text.replace('Mrs.', 'Mrs~')
    text = text.replace('Ms.', 'Ms~')
    text = text.replace('vs.', 'vs~')
    text = text.replace('e.g.', 'e~g~')
    text = text.replace('i.e.', 'i~e~')
    
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Restore abbreviations
    sentences = [s.replace('~', '.') for s in sentences]
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def extract_claims(text: str, min_words: int = 4) -> List[str]:
    """Extract factual claims from AI response"""
    sentences = extract_sentences(text)
    
    claims = []
    for sentence in sentences:
        # Skip very short sentences
        if len(sentence.split()) < min_words:
            continue
        
        # Skip questions
        if sentence.strip().endswith('?'):
            continue
        
        # Skip generic greetings/fillers
        filler_patterns = [
            r'^(hello|hi|hey|thank you|thanks|okay|ok|sure|great|good)\b',
            r'^(i see|i understand|i know|let me)',
            r'^(you can|you should|you may|you might)'
        ]
        
        is_filler = any(re.match(pattern, sentence.lower()) for pattern in filler_patterns)
        if is_filler and len(sentence.split()) < 8:
            continue
        
        claims.append(sentence.strip())
    
    return claims


# ==================== ENTITY EXTRACTION (NEW!) ====================

def extract_entities(text: str) -> Dict:
    """
    Extract important entities from text
    
    Returns dict with:
    - prices: List of normalized price mentions
    - hotels: List of hotel names
    - places: List of location names
    - numbers: List of important numbers
    - emails, urls, phone_numbers
    """
    entities = {
        'prices': [],
        'hotels': [],
        'places': [],
        'numbers': [],
        'emails': [],
        'urls': [],
        'phone_numbers': []
    }
    
    # Extract prices (Rs, USD, $, ₹)
    price_patterns = [
        r'(?:Rs\.?|₹|USD|\$)\s*\d+(?:,\d{3})*(?:\.\d{2})?',
        r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:rupees|dollars|USD|Rs)',
    ]
    for pattern in price_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities['prices'].extend([normalize_price(m) for m in matches])
    
    # Extract hotel names
    hotel_patterns = [
        r'(?:Hotel|Mansion|Inn|Lodge|Resort)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
        r'[A-Z][a-z]+\s+(?:Hotel|Mansion|Inn|Lodge|Resort)',
    ]
    for pattern in hotel_patterns:
        matches = re.findall(pattern, text)
        entities['hotels'].extend(matches)
    
    # Extract location names (capitalized words)
    location_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    potential_locations = re.findall(location_pattern, text)
    stop_words = {'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'Yes', 'No'}
    entities['places'] = [loc for loc in potential_locations if loc not in stop_words]
    
    # Extract numbers
    number_pattern = r'\b\d+(?:[.,]\d+)*\b'
    entities['numbers'] = re.findall(number_pattern, text)
    
    # Extract emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    entities['emails'] = re.findall(email_pattern, text)
    
    # Extract URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    entities['urls'] = re.findall(url_pattern, text)
    
    # Extract phone numbers
    phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}'
    entities['phone_numbers'] = re.findall(phone_pattern, text)
    
    return entities


def normalize_price(price_str: str) -> str:
    """
    Normalize price strings for comparison
    
    Examples:
    - "Rs 3000" -> "3000"
    - "₹3,000" -> "3000"
    - "$50" -> "50"
    """
    normalized = re.sub(r'[Rs₹$USD,.\s]', '', price_str, flags=re.IGNORECASE)
    return normalized


def entity_overlap_score(claim_entities: Dict, context_entities: Dict) -> float:
    """
    Calculate overlap score between entities in claim and context
    
    Returns score between 0 and 1
    """
    scores = []
    
    # Check price overlap
    if claim_entities['prices'] and context_entities['prices']:
        claim_prices = set(claim_entities['prices'])
        context_prices = set(context_entities['prices'])
        overlap = len(claim_prices & context_prices)
        total = len(claim_prices)
        scores.append(overlap / total if total > 0 else 0)
    
    # Check hotel overlap
    if claim_entities['hotels'] and context_entities['hotels']:
        claim_hotels = set(h.lower() for h in claim_entities['hotels'])
        context_hotels = set(h.lower() for h in context_entities['hotels'])
        overlap = len(claim_hotels & context_hotels)
        total = len(claim_hotels)
        scores.append(overlap / total if total > 0 else 0)
    
    # Check number overlap
    if claim_entities['numbers'] and context_entities['numbers']:
        claim_nums = set(claim_entities['numbers'])
        context_nums = set(context_entities['numbers'])
        overlap = len(claim_nums & context_nums)
        total = len(claim_nums)
        scores.append(overlap / total if total > 0 else 0)
    
    # Return average overlap score
    return sum(scores) / len(scores) if scores else 0.0


# ==================== TEXT CHUNKING ====================

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    """Split text into overlapping chunks for better semantic matching"""
    sentences = extract_sentences(text)
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # If sentence alone exceeds chunk_size, add it as its own chunk
        if sentence_tokens > chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            chunks.append(sentence)
            continue
        
        # If adding this sentence would exceed chunk_size
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Keep last few sentences for overlap
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_chunk):
                s_tokens = count_tokens(s)
                if overlap_tokens + s_tokens <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                else:
                    break
            
            current_chunk = overlap_sentences
            current_tokens = overlap_tokens
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def chunk_contexts(context_texts: List[str], chunk_size: int = 150) -> List[str]:
    """Chunk multiple context texts for better grounding detection"""
    all_chunks = []
    
    for text in context_texts:
        chunks = chunk_text(text, chunk_size=chunk_size)
        all_chunks.extend(chunks)
    
    return all_chunks


# ==================== FUZZY MATCHING ====================

def fuzzy_match_score(text1: str, text2: str) -> float:
    """Calculate fuzzy matching score using word overlap"""
    # Normalize texts
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'should', 'could', 'may', 'might', 'can', 'to', 'of',
                  'in', 'for', 'on', 'at', 'by', 'with', 'from', 'as'}
    
    words1 = words1 - stop_words
    words2 = words2 - stop_words
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


# ==================== MULTI-LEVEL GROUNDING ====================

def is_grounded_multi_level(claim: str, 
                            context_texts: List[str],
                            context_embeddings: np.ndarray,
                            claim_embedding: np.ndarray,
                            high_threshold: float = 0.70,
                            medium_threshold: float = 0.60,
                            fuzzy_threshold: float = 0.3) -> Tuple[bool, float, str]:
    """
    Multi-level grounding check with semantic, fuzzy, and entity matching
    
    Returns:
        Tuple of (is_grounded, best_score, matching_context)
    """
    if len(context_embeddings) == 0:
        return False, 0.0, ""
    
    # Calculate semantic similarities
    similarities = cosine_similarity_batch(claim_embedding, context_embeddings)
    best_idx = int(np.argmax(similarities))
    best_semantic_score = float(similarities[best_idx])
    best_context = context_texts[best_idx]
    
    # Level 1: Strong semantic match
    if best_semantic_score >= high_threshold:
        return True, best_semantic_score, best_context
    
    # Extract entities from claim
    claim_entities = extract_entities(claim)
    
    # Level 2: Check entity overlap for ALL contexts
    for idx, context in enumerate(context_texts):
        context_entities = extract_entities(context)
        entity_score = entity_overlap_score(claim_entities, context_entities)
        
        # Strong entity overlap = grounded
        if entity_score >= 0.70:
            combined_score = (similarities[idx] * 0.5) + (entity_score * 0.5)
            return True, combined_score, context
    
    # Level 3: Medium semantic + fuzzy match
    if best_semantic_score >= medium_threshold:
        fuzzy_score = fuzzy_match_score(claim, best_context)
        if fuzzy_score >= fuzzy_threshold:
            combined_score = (best_semantic_score * 0.7) + (fuzzy_score * 0.3)
            return True, combined_score, best_context
    
    # Level 4: Check if any context has high fuzzy match
    for idx, context in enumerate(context_texts):
        fuzzy_score = fuzzy_match_score(claim, context)
        if fuzzy_score >= 0.4:
            combined_score = (similarities[idx] * 0.6) + (fuzzy_score * 0.4)
            if combined_score >= medium_threshold:
                return True, combined_score, context
    
    # Not grounded
    return False, best_semantic_score, best_context


# ==================== TOKEN COUNTING & COST ====================

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken"""
    if not text:
        return 0
    
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return max(1, len(text) // 4)


def calculate_cost(input_tokens: int, 
                   output_tokens: int, 
                   api: str = "groq") -> float:
    """Calculate API cost based on token usage"""
    if api.lower() == "groq":
        input_cost_per_1m = float(os.getenv('GROQ_COST_PER_1M_INPUT_TOKENS') or '0.0')
        output_cost_per_1m = float(os.getenv('GROQ_COST_PER_1M_OUTPUT_TOKENS') or '0.0')
    else:
        input_cost_per_1m = 0.15
        output_cost_per_1m = 0.60
    
    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    
    return input_cost + output_cost


def calculate_embedding_cost(num_tokens: int) -> float:
    """Calculate embedding generation cost"""
    cost_per_1k = float(os.getenv('EMBEDDING_COST_PER_1K_TOKENS') or '0.0')
    return (num_tokens / 1000) * cost_per_1k


# ==================== UTILITIES ====================

def find_most_similar(query: str, 
                      documents: List[str], 
                      top_k: int = 3) -> List[Tuple[int, float, str]]:
    """Find most similar documents to a query"""
    if not documents:
        return []
    
    query_embedding = get_embeddings([query])[0]
    doc_embeddings = get_embeddings(documents)
    
    similarities = cosine_similarity_batch(query_embedding, doc_embeddings)
    
    top_k = min(top_k, len(similarities))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = [
        (int(idx), float(similarities[idx]), documents[idx])
        for idx in top_indices
    ]
    
    return results


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters"""
    if not text:
        return ""
    
    text = ' '.join(text.split())
    text = re.sub(r'[^\w\s.,!?;:\-\']', '', text)
    
    return text.strip()


def format_score(score: float, precision: int = 2) -> str:
    """Format score for display"""
    return f"{score:.{precision}f}"


def format_percentage(score: float, precision: int = 1) -> str:
    """Format score as percentage"""
    return f"{score * 100:.{precision}f}%"


# ==================== TOKEN COUNTER ====================

class TokenCounter:
    """Track token usage across multiple API calls"""
    
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.embedding_tokens = 0
    
    def add_llm_call(self, input_tokens: int, output_tokens: int):
        """Add tokens from an LLM API call"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
    
    def add_embedding_call(self, tokens: int):
        """Add tokens from an embedding API call"""
        self.embedding_tokens += tokens
    
    def get_total_cost(self) -> float:
        """Calculate total cost"""
        llm_cost = calculate_cost(self.input_tokens, self.output_tokens)
        embedding_cost = calculate_embedding_cost(self.embedding_tokens)
        return llm_cost + embedding_cost
    
    def reset(self):
        """Reset counters"""
        self.input_tokens = 0
        self.output_tokens = 0
        self.embedding_tokens = 0
    
    def summary(self) -> Dict[str, int]:
        """Get summary of token usage"""
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'embedding_tokens': self.embedding_tokens,
            'total_tokens': self.input_tokens + self.output_tokens + self.embedding_tokens
        }


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing complete utility functions...")
    print("=" * 60)
    
    # Test entity extraction
    print("\n1. Testing Entity Extraction:")
    test_text = "Gopal Mansion costs Rs 800 per night. Call us at +91-9867441589."
    entities = extract_entities(test_text)
    print(f"   Text: {test_text}")
    print(f"   Prices: {entities['prices']}")
    print(f"   Hotels: {entities['hotels']}")
    print(f"   Phone: {entities['phone_numbers']}")
    
    # Test entity overlap
    print("\n2. Testing Entity Overlap:")
    claim = "Hotel costs Rs 800"
    context = "An AC room is Rs 800 per night at the hotel"
    claim_ent = extract_entities(claim)
    context_ent = extract_entities(context)
    overlap = entity_overlap_score(claim_ent, context_ent)
    print(f"   Claim: {claim}")
    print(f"   Context: {context}")
    print(f"   Overlap: {overlap:.2f}")
    
    # Test multi-level grounding
    print("\n3. Testing Multi-Level Grounding:")
    claim = "Gopal Mansion is Rs 800 per night"
    contexts = [
        "The weather is nice today.",
        "An airconditioned room at Gopal Mansion costs Rs 800 per night."
    ]
    
    claim_emb = get_embeddings([claim])[0]
    context_embs = get_embeddings(contexts)
    
    is_grounded, score, match = is_grounded_multi_level(
        claim, contexts, context_embs, claim_emb
    )
    
    print(f"   Claim: {claim}")
    print(f"   Grounded: {is_grounded}")
    print(f"   Score: {score:.3f}")
    print(f"   Match: {match[:50]}...")
    
    print("\n" + "=" * 60)
    print("✨ All utility functions working!")