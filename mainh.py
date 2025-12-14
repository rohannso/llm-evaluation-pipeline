"""
Standalone Hybrid LLM Evaluation Script
Uses 30% LLM + 70% Semantic for improved accuracy

Usage:
    python evaluate_hybrid.py chat.json context.json
    python evaluate_hybrid.py chat.json context.json output.json
    python evaluate_hybrid.py --folder data/
    python evaluate_hybrid.py --compare chat.json context.json
"""

import sys
import json
import os
import glob
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv

# Import utilities
from utils import (
    get_embeddings,
    cosine_similarity,
    extract_claims,
    chunk_contexts,
    is_grounded_multi_level,
    count_tokens,
    TokenCounter,
    format_score,
    extract_entities,
    entity_overlap_score
)

load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Warning: groq library not installed. Install with: pip install groq")


class HybridEvaluator:
    """
    Hybrid LLM + Semantic Evaluation System
    Target: 30% LLM usage for improved accuracy
    """
    
    def __init__(self):
        """Initialize hybrid evaluator"""
        self.token_counter = TokenCounter()
        self.config = self._load_config()
        
        # LLM usage tracking
        self.stats = {
            'relevance_llm': 0,
            'claim_extraction_llm': 0,
            'grounding_llm': 0,
            'total_llm': 0,
            'total_turns': 0
        }
        
        # Initialize Groq
        self.groq_client = None
        if GROQ_AVAILABLE and self.config['use_hybrid']:
            api_key = os.getenv('GROQ_API_KEY')
            if api_key and api_key != 'your_groq_api_key_here':
                self.groq_client = Groq(api_key=api_key)
                print("✅ Groq LLM initialized")
            else:
                print("⚠️  Groq API key missing. Add to .env file.")
                print("   Falling back to semantic-only mode.")
                self.config['use_hybrid'] = False
        
        mode = "Hybrid (30% LLM)" if self.config['use_hybrid'] else "Semantic-Only"
        print(f"📊 Evaluation Mode: {mode}")
        print(f"   Model: {self.config['groq_model']}")
    
    def _load_config(self) -> Dict:
        """Load configuration"""
        return {
            'use_hybrid': os.getenv('USE_LLM_HYBRID', 'true').lower() == 'true',
            'groq_model': os.getenv('GROQ_MODEL', 'llama-3.1-70b-versatile'),
            'relevance_threshold': float(os.getenv('LLM_RELEVANCE_THRESHOLD', '0.50')),
            'claim_complexity': int(os.getenv('LLM_CLAIM_COMPLEXITY_THRESHOLD', '50')),
            'grounding_threshold': float(os.getenv('LLM_GROUNDING_THRESHOLD', '0.65')),
            'grounding_high': float(os.getenv('GROUNDING_HIGH_THRESHOLD', '0.70')),
            'grounding_medium': float(os.getenv('GROUNDING_MEDIUM_THRESHOLD', '0.60')),
            'fuzzy_threshold': float(os.getenv('FUZZY_MATCH_THRESHOLD', '0.30')),
            'chunk_size': int(os.getenv('CONTEXT_CHUNK_SIZE', '150')),
            'verbose': os.getenv('VERBOSE', 'true').lower() == 'true'
        }
    
    # ==================== RELEVANCE EVALUATION ====================
    
    def evaluate_relevance(self, query: str, response: str) -> Dict:
        """Hybrid relevance evaluation"""
        start = time.time()
        
        # Always do semantic first (fast)
        query_emb = get_embeddings([query])[0]
        response_emb = get_embeddings([response])[0]
        
        self.token_counter.add_embedding_call(count_tokens(query) + count_tokens(response))
        
        semantic_score = cosine_similarity(query_emb, response_emb)
        
        # Decide if we need LLM
        use_llm = False
        if self.config['use_hybrid'] and self.groq_client:
            # Use LLM if ambiguous OR complex
            is_ambiguous = 0.3 < semantic_score < 0.7
            is_complex = count_tokens(query) > 30 or count_tokens(response) > 100
            use_llm = is_ambiguous or is_complex
        
        # Get LLM score if needed
        if use_llm:
            llm_score = self._llm_relevance(query, response)
            self.stats['relevance_llm'] += 1
            self.stats['total_llm'] += 1
            final_score = (llm_score * 0.7) + (semantic_score * 0.3)
            method = "hybrid"
        else:
            final_score = semantic_score
            method = "semantic"
        
        # Label
        if final_score >= 0.7:
            label = "high"
        elif final_score <= 0.3:
            label = "low"
        else:
            label = "medium"
        
        return {
            'score': final_score,
            'label': label,
            'method': method,
            'llm_used': use_llm,
            'latency_ms': (time.time() - start) * 1000
        }
    
    def _llm_relevance(self, query: str, response: str) -> float:
        """LLM-based relevance scoring"""
        prompt = f"""Rate how well this response answers the question (0.0 to 1.0).

Question: {query}
Response: {response}

Provide ONLY a JSON response:
{{"score": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""
        
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.config['groq_model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            
            self.token_counter.add_llm_call(
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens
            )
            
            text = completion.choices[0].message.content.strip()
            text = text.replace('```json', '').replace('```', '').strip()
            result = json.loads(text)
            return float(result.get('score', 0.5))
        except Exception as e:
            if self.config['verbose']:
                print(f"      ⚠️  LLM relevance error: {e}")
            return 0.5
    
    # ==================== CLAIM EXTRACTION ====================
    
    def extract_claims_hybrid(self, response: str) -> List[str]:
        """Hybrid claim extraction"""
        tokens = count_tokens(response)
        
        # Simple: use regex
        if tokens < self.config['claim_complexity']:
            return extract_claims(response)
        
        # Complex: use LLM
        if self.config['use_hybrid'] and self.groq_client:
            llm_claims = self._llm_extract_claims(response)
            if llm_claims:
                self.stats['claim_extraction_llm'] += 1
                self.stats['total_llm'] += 1
                return llm_claims
        
        # Fallback
        return extract_claims(response)
    
    def _llm_extract_claims(self, text: str) -> List[str]:
        """LLM-based claim extraction"""
        prompt = f"""Extract all factual claims from this text.

TEXT: {text}

Rules:
- Extract ONLY facts (not greetings, questions, or generic phrases)
- Each claim should be complete and standalone
- Include specific details (prices, names, procedures, etc.)
- Skip phrases like "I understand" or "Let me help"

Provide ONLY a JSON array:
["claim 1", "claim 2", ...]"""
        
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.config['groq_model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500
            )
            
            self.token_counter.add_llm_call(
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens
            )
            
            text = completion.choices[0].message.content.strip()
            text = text.replace('```json', '').replace('```', '').strip()
            claims = json.loads(text)
            return claims if isinstance(claims, list) else []
        except Exception as e:
            if self.config['verbose']:
                print(f"      ⚠️  LLM claim extraction error: {e}")
            return []
    
    # ==================== GROUNDING VERIFICATION ====================
    
    def verify_grounding(self, claim: str, context_chunks: List[str],
                        context_embeddings, claim_embedding) -> Tuple[bool, float]:
        """Hybrid grounding verification"""
        
        # First: semantic + fuzzy + entity matching
        is_grounded, semantic_score, best_match = is_grounded_multi_level(
            claim=claim,
            context_texts=context_chunks,
            context_embeddings=context_embeddings,
            claim_embedding=claim_embedding,
            high_threshold=self.config['grounding_high'],
            medium_threshold=self.config['grounding_medium'],
            fuzzy_threshold=self.config['fuzzy_threshold']
        )
        
        # High confidence: trust semantic
        if semantic_score >= 0.70:
            return True, semantic_score
        
        # Very low: trust semantic
        if semantic_score < 0.50:
            return False, semantic_score
        
        # Borderline: use LLM
        if self.config['use_hybrid'] and self.groq_client:
            llm_grounded = self._llm_verify_grounding(claim, best_match)
            self.stats['grounding_llm'] += 1
            self.stats['total_llm'] += 1
            
            if llm_grounded:
                return True, max(semantic_score, 0.75)
            else:
                return False, semantic_score
        
        # Fallback
        return is_grounded, semantic_score
    
    def _llm_verify_grounding(self, claim: str, context: str) -> bool:
        """LLM-based grounding verification"""
        prompt = f"""Is this CLAIM supported by the CONTEXT?

CLAIM: {claim}
CONTEXT: {context}

Answer YES if the context supports the claim (even if paraphrased).
Answer NO if the context contradicts or doesn't mention it.

Provide ONLY a JSON response:
{{"supported": true/false, "confidence": <0-1>, "reasoning": "<brief>"}}"""
        
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.config['groq_model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150
            )
            
            self.token_counter.add_llm_call(
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens
            )
            
            text = completion.choices[0].message.content.strip()
            text = text.replace('```json', '').replace('```', '').strip()
            result = json.loads(text)
            return result.get('supported', False)
        except Exception as e:
            if self.config['verbose']:
                print(f"      ⚠️  LLM grounding error: {e}")
            return False
    
    # ==================== HALLUCINATION DETECTION ====================
    
    def evaluate_hallucination(self, response: str, context_texts: List[str]) -> Dict:
        """Hybrid hallucination detection"""
        start = time.time()
        
        if not context_texts:
            return {
                'score': 0.0,
                'hallucinated_claims': [],
                'total_claims': 0,
                'grounded_claims': 0,
                'latency_ms': (time.time() - start) * 1000
            }
        
        # Extract claims (hybrid)
        claims = self.extract_claims_hybrid(response)
        
        if not claims:
            return {
                'score': 0.0,
                'hallucinated_claims': [],
                'total_claims': 0,
                'grounded_claims': 0,
                'latency_ms': (time.time() - start) * 1000
            }
        
        # Chunk contexts
        chunks = chunk_contexts(context_texts, self.config['chunk_size'])
        
        if not chunks:
            return {
                'score': 1.0,
                'hallucinated_claims': claims,
                'total_claims': len(claims),
                'grounded_claims': 0,
                'latency_ms': (time.time() - start) * 1000
            }
        
        # Generate embeddings
        claim_embs = get_embeddings(claims)
        context_embs = get_embeddings(chunks)
        
        tokens = sum(count_tokens(c) for c in claims + chunks)
        self.token_counter.add_embedding_call(tokens)
        
        # Check grounding (hybrid)
        hallucinated = []
        grounded_count = 0
        
        for i, claim in enumerate(claims):
            is_grounded, score = self.verify_grounding(
                claim, chunks, context_embs, claim_embs[i]
            )
            
            if is_grounded:
                grounded_count += 1
            else:
                hallucinated.append(claim)
        
        hallucination_score = len(hallucinated) / len(claims)
        
        return {
            'score': hallucination_score,
            'hallucinated_claims': hallucinated,
            'total_claims': len(claims),
            'grounded_claims': grounded_count,
            'latency_ms': (time.time() - start) * 1000
        }
    
    # ==================== MAIN EVALUATION ====================
    
    def evaluate_conversation(self, chat_data: Dict, context_data: Dict) -> Dict:
        """Evaluate entire conversation"""
        print(f"\n🔍 Starting hybrid evaluation...")
        
        # Extract data
        ai_turns = [t for t in chat_data['conversation_turns'] if t['role'] == 'AI/Chatbot']
        all_turns = chat_data['conversation_turns']
        
        context_texts = []
        for vector in context_data['data']['vector_data']:
            if 'text' in vector and vector['text']:
                context_texts.append(vector['text'].strip())
        
        print(f"   AI turns: {len(ai_turns)}")
        print(f"   Context docs: {len(context_texts)}")
        
        # Evaluate each turn
        turn_results = []
        
        for ai_turn in ai_turns:
            self.stats['total_turns'] += 1
            
            # Find previous user turn
            prev_turn = None
            for t in all_turns:
                if t['turn'] == ai_turn['turn'] - 1 and t['role'] == 'User':
                    prev_turn = t
                    break
            
            print(f"\n   📝 Turn {ai_turn['turn']}...")
            
            # Evaluate relevance
            if prev_turn:
                rel_result = self.evaluate_relevance(prev_turn['message'], ai_turn['message'])
            else:
                rel_result = {'score': 1.0, 'label': 'N/A', 'method': 'none', 'llm_used': False, 'latency_ms': 0}
            
            # Evaluate hallucination
            hal_result = self.evaluate_hallucination(ai_turn['message'], context_texts)
            
            # Store results
            turn_results.append({
                'turn': ai_turn['turn'],
                'message': ai_turn['message'][:100] + '...' if len(ai_turn['message']) > 100 else ai_turn['message'],
                'user_query': prev_turn['message'][:100] + '...' if prev_turn and len(prev_turn['message']) > 100 else (prev_turn['message'] if prev_turn else 'No query'),
                'relevance_score': rel_result['score'],
                'relevance_label': rel_result['label'],
                'relevance_method': rel_result['method'],
                'hallucination_score': hal_result['score'],
                'hallucinated_claims': hal_result['hallucinated_claims'],
                'total_claims': hal_result['total_claims'],
                'grounded_claims': hal_result['grounded_claims'],
                'latency_ms': round(rel_result['latency_ms'] + hal_result['latency_ms'], 2),
                'cost_usd': round(self.token_counter.get_total_cost(), 6),
                'evaluation_timestamp': datetime.now().isoformat()
            })
            
            if self.config['verbose']:
                print(f"      Relevance: {format_score(rel_result['score'])} ({rel_result['method']})")
                print(f"      Hallucination: {format_score(hal_result['score'])} ({hal_result['grounded_claims']}/{hal_result['total_claims']} grounded)")
        
        # Calculate summary
        if turn_results:
            avg_rel = sum(t['relevance_score'] for t in turn_results) / len(turn_results)
            avg_hal = sum(t['hallucination_score'] for t in turn_results) / len(turn_results)
            total_lat = sum(t['latency_ms'] for t in turn_results)
        else:
            avg_rel = avg_hal = total_lat = 0.0
        
        total_cost = self.token_counter.get_total_cost()
        
        # Create report
        report = {
            'conversation_id': chat_data['chat_id'],
            'user_id': chat_data['user_id'],
            'evaluation_mode': 'hybrid_llm' if self.config['use_hybrid'] else 'semantic_only',
            'total_turns_evaluated': len(turn_results),
            'evaluation_summary': {
                'avg_relevance_score': round(avg_rel, 3),
                'avg_hallucination_rate': round(avg_hal, 3),
                'total_latency_ms': round(total_lat, 2),
                'total_cost_usd': round(total_cost, 6),
                'token_usage': self.token_counter.summary(),
                'llm_usage': self.get_llm_stats()
            },
            'turn_evaluations': turn_results,
            'evaluated_at': datetime.now().isoformat()
        }
        
        print(f"\n✅ Evaluation complete!")
        print(f"   Avg Relevance: {format_score(avg_rel)}")
        print(f"   Avg Hallucination: {format_score(avg_hal)}")
        print(f"   Total Latency: {total_lat:.0f}ms")
        print(f"   Total Cost: ${total_cost:.6f}")
        
        return report
    
    def get_llm_stats(self) -> Dict:
        """Get LLM usage statistics"""
        total_ops = self.stats['total_turns'] * 3  # Approx operations per turn
        llm_pct = (self.stats['total_llm'] / total_ops * 100) if total_ops > 0 else 0
        
        return {
            'total_llm_calls': self.stats['total_llm'],
            'relevance_calls': self.stats['relevance_llm'],
            'claim_extraction_calls': self.stats['claim_extraction_llm'],
            'grounding_calls': self.stats['grounding_llm'],
            'percentage': round(llm_pct, 1)
        }
    
    def print_stats(self):
        """Print detailed statistics"""
        stats = self.get_llm_stats()
        print(f"\n📊 LLM Usage Statistics:")
        print(f"   Total Turns: {self.stats['total_turns']}")
        print(f"   Total LLM Calls: {stats['total_llm_calls']}")
        print(f"   - Relevance: {stats['relevance_calls']}")
        print(f"   - Claim Extraction: {stats['claim_extraction_calls']}")
        print(f"   - Grounding: {stats['grounding_calls']}")
        print(f"   LLM Usage: ~{stats['percentage']:.1f}%")
    
    def load_data(self, chat_path: str, context_path: str) -> Tuple[Dict, Dict]:
        """Load JSON files"""
        with open(chat_path, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        with open(context_path, 'r', encoding='utf-8') as f:
            context_data = json.load(f)
        
        return chat_data, context_data
    
    def save_results(self, report: Dict, output_path: str):
        """Save evaluation report"""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main execution"""
    print("="*70)
    print("🚀 Hybrid LLM Evaluation Pipeline")
    print("="*70)
    
    args = sys.argv[1:]
    
    if not args or '--help' in args or '-h' in args:
        print(__doc__)
        return
    
    # Check for folder mode
    if '--folder' in args:
        idx = args.index('--folder')
        if idx + 1 >= len(args):
            print("❌ Error: --folder requires a path")
            return
        
        folder_path = args[idx + 1]
        batch_evaluate(folder_path)
        return
    
    # Single file evaluation
    if len(args) < 2:
        print("❌ Error: Requires chat and context files")
        print("Usage: python evaluate_hybrid.py chat.json context.json [output.json]")
        return
    
    chat_path = args[0]
    context_path = args[1]
    output_path = args[2] if len(args) > 2 else None
    
    # Check files exist
    if not os.path.exists(chat_path):
        print(f"❌ Chat file not found: {chat_path}")
        return
    
    if not os.path.exists(context_path):
        print(f"❌ Context file not found: {context_path}")
        return
    
    # Generate output path if needed
    if output_path is None:
        chat_name = Path(chat_path).stem
        output_dir = Path(chat_path).parent / 'results'
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"{chat_name}_hybrid_evaluation.json")
    
    # Run evaluation
    evaluator = HybridEvaluator()
    chat_data, context_data = evaluator.load_data(chat_path, context_path)
    report = evaluator.evaluate_conversation(chat_data, context_data)
    evaluator.save_results(report, output_path)
    evaluator.print_stats()
    
    print("\n" + "="*70)
    print("✨ Hybrid evaluation completed!")
    print("="*70)


def batch_evaluate(folder_path: str):
    """Batch evaluate all conversations in folder"""
    print(f"\n📁 Batch Evaluation: {folder_path}")
    
    # Find chat files
    chat_files = glob.glob(os.path.join(folder_path, '*chat*.json'))
    chat_files += glob.glob(os.path.join(folder_path, '*conversation*.json'))
    chat_files = list(set(chat_files))
    
    if not chat_files:
        print(f"❌ No chat files found in {folder_path}")
        return
    
    print(f"Found {len(chat_files)} conversation(s)")
    
    success = 0
    failed = 0
    
    evaluator = HybridEvaluator()
    
    for chat_path in chat_files:
        # Find context file
        chat_name = Path(chat_path).stem
        context_patterns = [
            chat_name.replace('chat', 'context') + '.json',
            chat_name.replace('conversation', 'context') + '.json',
            chat_name + '_context.json',
            chat_name + '_vectors.json'
        ]
        
        context_path = None
        for pattern in context_patterns:
            test_path = Path(chat_path).parent / pattern
            if test_path.exists():
                context_path = str(test_path)
                break
        
        if not context_path:
            print(f"\n⚠️  Skipping {Path(chat_path).name}: No context file found")
            failed += 1
            continue
        
        # Evaluate
        try:
            print(f"\n{'='*70}")
            print(f"Evaluating: {Path(chat_path).name}")
            print(f"{'='*70}")
            
            chat_data, context_data = evaluator.load_data(chat_path, context_path)
            report = evaluator.evaluate_conversation(chat_data, context_data)
            
            output_dir = Path(chat_path).parent / 'results'
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{chat_name}_hybrid.json"
            
            evaluator.save_results(report, str(output_path))
            success += 1
        except Exception as e:
            print(f"❌ Failed: {e}")
            failed += 1
    
    print(f"\n{'='*70}")
    print("📊 Batch Evaluation Complete")
    print(f"{'='*70}")
    print(f"✅ Successful: {success}")
    print(f"❌ Failed: {failed}")
    evaluator.print_stats()


if __name__ == "__main__":
    main()