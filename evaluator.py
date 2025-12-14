"""
LLM Evaluation Pipeline - Main Script (Improved)
Evaluates LLM responses for:
- Relevance & Completeness
- Hallucination / Factual Accuracy (with multi-level grounding)
- Latency & Cost

Author: BeyondChats Assignment
Date: December 2024
"""

import json
import os
import time
from typing import List, Dict, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Import our improved utilities
from utils import (
    get_embeddings,
    cosine_similarity,
    extract_claims,
    chunk_contexts,
    is_grounded_multi_level,
    count_tokens,
    TokenCounter,
    format_score
)

# Load environment variables
load_dotenv()

# Try to import Groq (optional)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Groq library not installed. LLM judge will be disabled.")


class EvaluationPipeline:
    """Main evaluation pipeline for LLM responses"""
    
    def __init__(self):
        """Initialize the evaluation pipeline"""
        self.token_counter = TokenCounter()
        self.config = self._load_config()
        
        # Initialize Groq client if LLM judge is enabled
        self.groq_client = None
        if self.config['use_llm_judge'] and GROQ_AVAILABLE:
            api_key = os.getenv('GROQ_API_KEY')
            if api_key and api_key != 'your_groq_api_key_here':
                self.groq_client = Groq(api_key=api_key)
                print("✅ Groq client initialized")
            else:
                print("⚠️  Groq API key not found. LLM judge disabled.")
                self.config['use_llm_judge'] = False
        
        print(f"📊 Evaluation Pipeline initialized")
        print(f"   Relevance thresholds: {self.config['relevance_low']:.2f} - {self.config['relevance_high']:.2f}")
        print(f"   Grounding thresholds: High={self.config['grounding_high']:.2f}, Medium={self.config['grounding_medium']:.2f}")
        print(f"   Context chunk size: {self.config['chunk_size']} tokens")
        print(f"   LLM Judge: {'Enabled' if self.config['use_llm_judge'] else 'Disabled'}")
    
    def _load_config(self) -> Dict:
        """Load configuration from environment with safe defaults"""
        
        def safe_float(key: str, default: float) -> float:
            """Safely get float from environment"""
            try:
                value = os.getenv(key)
                return float(value) if value else default
            except (ValueError, TypeError):
                return default
        
        def safe_int(key: str, default: int) -> int:
            """Safely get int from environment"""
            try:
                value = os.getenv(key)
                return int(value) if value else default
            except (ValueError, TypeError):
                return default
        
        def safe_bool(key: str, default: bool) -> bool:
            """Safely get bool from environment"""
            try:
                value = os.getenv(key)
                return value.lower() == 'true' if value else default
            except (AttributeError, TypeError):
                return default
        
        def safe_str(key: str, default: str) -> str:
            """Safely get string from environment"""
            value = os.getenv(key)
            return value if value else default
        
        return {
            'use_llm_judge': safe_bool('USE_LLM_JUDGE', False),
            'relevance_high': safe_float('RELEVANCE_HIGH_THRESHOLD', 0.7),
            'relevance_low': safe_float('RELEVANCE_LOW_THRESHOLD', 0.3),
            'grounding_high': safe_float('GROUNDING_HIGH_THRESHOLD', 0.70),
            'grounding_medium': safe_float('GROUNDING_MEDIUM_THRESHOLD', 0.60),
            'fuzzy_threshold': safe_float('FUZZY_MATCH_THRESHOLD', 0.30),
            'chunk_size': safe_int('CONTEXT_CHUNK_SIZE', 150),
            'groq_model': safe_str('GROQ_MODEL', 'llama-3.1-8b-instant'),
            'verbose': safe_bool('VERBOSE', True)
        }
    
    def load_data(self, chat_path: str, context_path: str) -> Tuple[Dict, Dict]:
        """
        Load chat conversation and context vectors from JSON files
        
        Args:
            chat_path: Path to chat conversation JSON
            context_path: Path to context vectors JSON
            
        Returns:
            Tuple of (chat_data, context_data)
        """
        print(f"\n📂 Loading data...")
        
        try:
            with open(chat_path, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
            print(f"   ✅ Loaded chat conversation: {len(chat_data.get('conversation_turns', []))} turns")
        except Exception as e:
            print(f"   ❌ Error loading chat file: {e}")
            raise
        
        try:
            with open(context_path, 'r', encoding='utf-8') as f:
                context_data = json.load(f)
            
            # Handle different JSON structures
            if 'data' in context_data and 'vector_data' in context_data['data']:
                vector_count = len(context_data['data']['vector_data'])
            else:
                print("   ⚠️  Unexpected context JSON structure")
                vector_count = 0
            
            print(f"   ✅ Loaded context vectors: {vector_count} vectors")
        except Exception as e:
            print(f"   ❌ Error loading context file: {e}")
            raise
        
        return chat_data, context_data
    
    def extract_ai_responses(self, chat_data: Dict) -> List[Dict]:
        """
        Extract AI/Chatbot responses from conversation
        
        Args:
            chat_data: Chat conversation data
            
        Returns:
            List of AI response turns
        """
        ai_responses = [
            turn for turn in chat_data['conversation_turns']
            if turn['role'] == 'AI/Chatbot'
        ]
        return ai_responses
    
    def get_context_texts(self, context_data: Dict) -> List[str]:
        """
        Extract context texts from vector data
        
        Args:
            context_data: Context vectors data
            
        Returns:
            List of context text strings
        """
        context_texts = []
        for vector in context_data['data']['vector_data']:
            # Check if 'text' key exists and has content
            if 'text' in vector and vector['text']:
                text = vector['text'].strip()
                if text:  # Only add non-empty texts
                    context_texts.append(text)
        
        if not context_texts:
            print("⚠️  Warning: No valid context texts found!")
        else:
            print(f"   ✅ Found {len(context_texts)} context documents")
        
        return context_texts
    
    def evaluate_relevance(self, user_query: str, ai_response: str) -> Dict:
        """
        Evaluate relevance and completeness of AI response
        
        Args:
            user_query: User's question
            ai_response: AI's response
            
        Returns:
            Dict with relevance score and details
        """
        start_time = time.time()
        
        # Generate embeddings
        query_embedding = get_embeddings([user_query])[0]
        response_embedding = get_embeddings([ai_response])[0]
        
        # Track tokens
        query_tokens = count_tokens(user_query)
        response_tokens = count_tokens(ai_response)
        self.token_counter.add_embedding_call(query_tokens + response_tokens)
        
        # Calculate semantic similarity
        similarity = cosine_similarity(query_embedding, response_embedding)
        
        # Determine relevance based on thresholds
        if similarity >= self.config['relevance_high']:
            relevance_label = "high"
            relevance_score = similarity
        elif similarity <= self.config['relevance_low']:
            relevance_label = "low"
            relevance_score = similarity
        else:
            # Edge case - use LLM judge if enabled
            relevance_label = "medium"
            if self.config['use_llm_judge'] and self.groq_client:
                relevance_score = self._llm_judge_relevance(user_query, ai_response)
            else:
                relevance_score = similarity
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            'score': relevance_score,
            'label': relevance_label,
            'latency_ms': latency
        }
    
    def _llm_judge_relevance(self, query: str, response: str) -> float:
        """
        Use LLM to judge relevance (for edge cases)
        
        Args:
            query: User query
            response: AI response
            
        Returns:
            Relevance score (0-1)
        """
        prompt = f"""Rate how well this response answers the question. Return ONLY a number between 0 and 1.

Question: {query}

Response: {response}

Relevance score (0-1):"""
        
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.config['groq_model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            # Track tokens
            self.token_counter.add_llm_call(
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens
            )
            
            # Parse score
            score_text = completion.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"   ⚠️  LLM judge error: {e}")
            return 0.5  # Fallback to neutral score
    
    def evaluate_hallucination(self, ai_response: str, context_texts: List[str]) -> Dict:
        """
        Detect hallucinations using improved multi-level grounding detection
        
        Args:
            ai_response: AI's response
            context_texts: Available context texts
            
        Returns:
            Dict with hallucination score and detected claims
        """
        start_time = time.time()
        
        # Check if we have context
        if not context_texts:
            print("   ⚠️  No context texts available for hallucination check")
            return {
                'score': 0.0,
                'hallucinated_claims': [],
                'total_claims': 0,
                'grounded_claims': 0,
                'latency_ms': (time.time() - start_time) * 1000
            }
        
        # Extract claims from AI response
        claims = extract_claims(ai_response)
        
        if not claims:
            # No factual claims to check (e.g., just a greeting)
            return {
                'score': 0.0,
                'hallucinated_claims': [],
                'total_claims': 0,
                'grounded_claims': 0,
                'latency_ms': (time.time() - start_time) * 1000
            }
        
        # Chunk contexts for better matching
        context_chunks = chunk_contexts(context_texts, self.config['chunk_size'])
        
        if not context_chunks:
            print("   ⚠️  No context chunks generated")
            return {
                'score': 1.0,
                'hallucinated_claims': claims,
                'total_claims': len(claims),
                'grounded_claims': 0,
                'latency_ms': (time.time() - start_time) * 1000
            }
        
        if self.config['verbose']:
            print(f"      Checking {len(claims)} claims against {len(context_chunks)} context chunks...")
        
        # Generate embeddings
        claim_embeddings = get_embeddings(claims)
        context_embeddings = get_embeddings(context_chunks)
        
        # Track tokens
        total_tokens = sum(count_tokens(c) for c in claims)
        total_tokens += sum(count_tokens(c) for c in context_chunks)
        self.token_counter.add_embedding_call(total_tokens)
        
        # Check grounding for each claim using multi-level approach
        hallucinated_claims = []
        grounded_count = 0
        grounding_details = []
        
        for i, claim in enumerate(claims):
            claim_embedding = claim_embeddings[i]
            
            # Use improved multi-level grounding check
            is_grounded, score, best_match = is_grounded_multi_level(
                claim=claim,
                context_texts=context_chunks,
                context_embeddings=context_embeddings,
                claim_embedding=claim_embedding,
                high_threshold=self.config['grounding_high'],
                medium_threshold=self.config['grounding_medium'],
                fuzzy_threshold=self.config['fuzzy_threshold']
            )
            
            if is_grounded:
                grounded_count += 1
                if self.config['verbose']:
                    grounding_details.append({
                        'claim': claim[:60] + '...' if len(claim) > 60 else claim,
                        'grounded': True,
                        'score': round(score, 3)
                    })
            else:
                hallucinated_claims.append(claim)
                if self.config['verbose']:
                    grounding_details.append({
                        'claim': claim[:60] + '...' if len(claim) > 60 else claim,
                        'grounded': False,
                        'score': round(score, 3),
                        'reason': 'Low similarity to all contexts'
                    })
        
        # Print detailed grounding info if verbose
        if self.config['verbose'] and grounding_details:
            print(f"      Grounding breakdown:")
            for detail in grounding_details[:3]:  # Show first 3
                status = "✓ Grounded" if detail['grounded'] else "✗ Hallucinated"
                print(f"        {status} (score: {detail['score']:.3f}): {detail['claim']}")
        
        # Calculate hallucination score (percentage of ungrounded claims)
        hallucination_score = len(hallucinated_claims) / len(claims)
        
        latency = (time.time() - start_time) * 1000
        
        return {
            'score': hallucination_score,
            'hallucinated_claims': hallucinated_claims,
            'total_claims': len(claims),
            'grounded_claims': grounded_count,
            'latency_ms': latency
        }
    
    def evaluate_turn(self, turn: Dict, context_texts: List[str], 
                     previous_turn: Dict = None) -> Dict:
        """
        Evaluate a single conversation turn
        
        Args:
            turn: AI response turn
            context_texts: Available context texts
            previous_turn: Previous user turn (for relevance check)
            
        Returns:
            Evaluation results for this turn
        """
        start_time = time.time()
        
        ai_message = turn['message']
        
        # Get user query from previous turn
        user_query = previous_turn['message'] if previous_turn else "No user query"
        
        # Evaluate relevance
        if previous_turn and previous_turn['role'] == 'User':
            relevance_result = self.evaluate_relevance(user_query, ai_message)
        else:
            relevance_result = {'score': 1.0, 'label': 'N/A', 'latency_ms': 0}
        
        # Evaluate hallucination
        hallucination_result = self.evaluate_hallucination(ai_message, context_texts)
        
        # Calculate total latency and cost
        total_latency = (time.time() - start_time) * 1000
        turn_cost = self.token_counter.get_total_cost()
        
        return {
            'turn': turn['turn'],
            'message': ai_message[:100] + '...' if len(ai_message) > 100 else ai_message,
            'user_query': user_query[:100] + '...' if len(user_query) > 100 else user_query,
            'relevance_score': relevance_result['score'],
            'relevance_label': relevance_result['label'],
            'hallucination_score': hallucination_result['score'],
            'hallucinated_claims': hallucination_result['hallucinated_claims'],
            'total_claims': hallucination_result['total_claims'],
            'grounded_claims': hallucination_result['grounded_claims'],
            'latency_ms': round(total_latency, 2),
            'cost_usd': round(turn_cost, 6),
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def evaluate_conversation(self, chat_data: Dict, context_data: Dict) -> Dict:
        """
        Evaluate entire conversation
        
        Args:
            chat_data: Chat conversation data
            context_data: Context vectors data
            
        Returns:
            Complete evaluation results
        """
        print(f"\n🔍 Starting evaluation...")
        
        # Extract data
        ai_responses = self.extract_ai_responses(chat_data)
        context_texts = self.get_context_texts(context_data)
        all_turns = chat_data['conversation_turns']
        
        print(f"   Evaluating {len(ai_responses)} AI responses...")
        
        # Evaluate each AI response
        turn_evaluations = []
        
        for ai_turn in ai_responses:
            # Find the previous user turn
            turn_index = ai_turn['turn']
            previous_turn = None
            
            for t in all_turns:
                if t['turn'] == turn_index - 1 and t['role'] == 'User':
                    previous_turn = t
                    break
            
            # Evaluate this turn
            print(f"\n   📝 Evaluating turn {ai_turn['turn']}...")
            evaluation = self.evaluate_turn(ai_turn, context_texts, previous_turn)
            turn_evaluations.append(evaluation)
            
            if self.config['verbose']:
                print(f"      Relevance: {format_score(evaluation['relevance_score'])} ({evaluation['relevance_label']})")
                print(f"      Hallucination: {format_score(evaluation['hallucination_score'])} ({evaluation['grounded_claims']}/{evaluation['total_claims']} claims grounded)")
        
        # Calculate summary statistics
        if turn_evaluations:
            avg_relevance = sum(e['relevance_score'] for e in turn_evaluations) / len(turn_evaluations)
            avg_hallucination = sum(e['hallucination_score'] for e in turn_evaluations) / len(turn_evaluations)
            total_latency = sum(e['latency_ms'] for e in turn_evaluations)
            total_cost = self.token_counter.get_total_cost()
        else:
            avg_relevance = 0.0
            avg_hallucination = 0.0
            total_latency = 0.0
            total_cost = 0.0
        
        # Create final report
        report = {
            'conversation_id': chat_data['chat_id'],
            'user_id': chat_data['user_id'],
            'total_turns_evaluated': len(turn_evaluations),
            'evaluation_summary': {
                'avg_relevance_score': round(avg_relevance, 3),
                'avg_hallucination_rate': round(avg_hallucination, 3),
                'total_latency_ms': round(total_latency, 2),
                'total_cost_usd': round(total_cost, 6),
                'token_usage': self.token_counter.summary()
            },
            'turn_evaluations': turn_evaluations,
            'evaluated_at': datetime.now().isoformat()
        }
        
        print(f"\n✅ Evaluation complete!")
        print(f"   Average Relevance: {format_score(avg_relevance)}")
        print(f"   Average Hallucination Rate: {format_score(avg_hallucination)}")
        print(f"   Total Latency: {total_latency:.0f}ms")
        print(f"   Total Cost: ${total_cost:.6f}")
        
        return report
    
    def save_results(self, report: Dict, output_path: str):
        """
        Save evaluation results to JSON file
        
        Args:
            report: Evaluation report
            output_path: Output file path
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main execution function"""
    print("="*60)
    print("🚀 LLM Evaluation Pipeline (Improved)")
    print("="*60)
    
    # Check if .env exists
    if not os.path.exists('.env'):
        print("\n⚠️  Warning: .env file not found!")
        print("   Using default configuration...")
        print("   (You can create .env from .env.example for custom settings)\n")
    
    # File paths
    chat_path = 'data/sample-chat-conversation-01.json'
    context_path = 'data/sample_context_vectors-0.json'
    output_path = os.getenv('OUTPUT_FILE') or 'data/evaluation_results.json'
    
    # Check if input files exist
    if not os.path.exists(chat_path):
        print(f"❌ Error: Chat file not found: {chat_path}")
        print(f"   Please copy your sample JSON files to the data/ folder")
        return
    
    if not os.path.exists(context_path):
        print(f"❌ Error: Context file not found: {context_path}")
        print(f"   Please copy your sample JSON files to the data/ folder")
        return
    
    # Initialize pipeline
    pipeline = EvaluationPipeline()
    
    # Load data
    chat_data, context_data = pipeline.load_data(chat_path, context_path)
    
    # Run evaluation
    report = pipeline.evaluate_conversation(chat_data, context_data)
    
    # Save results
    pipeline.save_results(report, output_path)
    
    print("\n" + "="*60)
    print("✨ Evaluation pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()