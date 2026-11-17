"""
Vast Multi-Strategy Chunking Engine
Supports: Fixed-Size, Sliding Window with Overlap, Semantic Chunking,
Metadata-Aware Chunking, and Hierarchical/Recursive Chunking.
"""
import re
from typing import List, Dict, Any, Literal

class ChunkStrategy:
    FIXED_SIZE = "fixed_size"
    SLIDING_WINDOW = "sliding_window"
    SEMANTIC = "semantic"
    METADATA_AWARE = "metadata_aware"
    HIERARCHICAL = "hierarchical"

class ChunkingEngine:
    def __init__(self):
        pass

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        # Clean regex to split on sentence boundaries for English and Indic punctuation (।, ?, !)
        sentences = re.split(r'(?<=[.!?।])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_fixed_size(self, doc: Dict[str, Any], chunk_size: int = 150) -> List[Dict[str, Any]]:
        text = doc.get('content', '')
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append({
                'chunk_id': f"{doc['doc_id']}_fixed_{i // chunk_size}",
                'doc_id': doc['doc_id'],
                'text': chunk_text,
                'strategy': ChunkStrategy.FIXED_SIZE,
                'metadata': {
                    'title': doc.get('title', ''),
                    'language': doc.get('language', 'en'),
                    'start_token': i,
                    'end_token': i + len(chunk_words)
                }
            })
        return chunks

    def chunk_sliding_window(self, doc: Dict[str, Any], chunk_size: int = 120, overlap: int = 40) -> List[Dict[str, Any]]:
        text = doc.get('content', '')
        words = text.split()
        chunks = []
        stride = max(1, chunk_size - overlap)
        chunk_idx = 0
        for i in range(0, len(words), stride):
            chunk_words = words[i:i + chunk_size]
            if not chunk_words:
                continue
            chunk_text = ' '.join(chunk_words)
            chunks.append({
                'chunk_id': f"{doc['doc_id']}_sliding_{chunk_idx}",
                'doc_id': doc['doc_id'],
                'text': chunk_text,
                'strategy': ChunkStrategy.SLIDING_WINDOW,
                'metadata': {
                    'title': doc.get('title', ''),
                    'language': doc.get('language', 'en'),
                    'overlap_tokens': overlap,
                    'window_start': i,
                    'window_end': i + len(chunk_words)
                }
            })
            chunk_idx += 1
            if i + chunk_size >= len(words):
                break
        return chunks

    def chunk_semantic(self, doc: Dict[str, Any], max_chunk_sentences: int = 3) -> List[Dict[str, Any]]:
        text = doc.get('content', '')
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        chunk_idx = 0

        for s in sentences:
            current_chunk.append(s)
            # Break on thematic or sentence limit
            if len(current_chunk) >= max_chunk_sentences:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'chunk_id': f"{doc['doc_id']}_semantic_{chunk_idx}",
                    'doc_id': doc['doc_id'],
                    'text': chunk_text,
                    'strategy': ChunkStrategy.SEMANTIC,
                    'metadata': {
                        'title': doc.get('title', ''),
                        'language': doc.get('language', 'en'),
                        'sentence_count': len(current_chunk),
                        'topic': doc.get('topic', 'General')
                    }
                })
                chunk_idx += 1
                current_chunk = []

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'chunk_id': f"{doc['doc_id']}_semantic_{chunk_idx}",
                'doc_id': doc['doc_id'],
                'text': chunk_text,
                'strategy': ChunkStrategy.SEMANTIC,
                'metadata': {
                    'title': doc.get('title', ''),
                    'language': doc.get('language', 'en'),
                    'sentence_count': len(current_chunk),
                    'topic': doc.get('topic', 'General')
                }
            })
        return chunks

    def chunk_metadata_aware(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Embeds document structure, language tag, topic, and section headers
        directly into the indexed chunk representation.
        """
        text = doc.get('content', '')
        sentences = self._split_into_sentences(text)
        sections = doc.get('section_headers', ['General'])
        chunks = []
        
        # Group sentences by section
        group_size = max(1, len(sentences) // len(sections)) if sections else 2
        for idx, sec in enumerate(sections):
            start = idx * group_size
            end = start + group_size if idx < len(sections) - 1 else len(sentences)
            sec_sentences = sentences[start:end]
            if not sec_sentences:
                continue
            
            raw_text = ' '.join(sec_sentences)
            # Context enriched chunk
            enriched_text = f"[Document: {doc.get('title', '')} | Topic: {doc.get('topic', '')} | Language: {doc.get('language', '')} | Section: {sec}]\n{raw_text}"
            
            chunks.append({
                'chunk_id': f"{doc['doc_id']}_meta_{idx}",
                'doc_id': doc['doc_id'],
                'text': enriched_text,
                'raw_text': raw_text,
                'strategy': ChunkStrategy.METADATA_AWARE,
                'metadata': {
                    'title': doc.get('title', ''),
                    'topic': doc.get('topic', ''),
                    'section': sec,
                    'language': doc.get('language', 'en')
                }
            })
        return chunks

    def chunk_hierarchical(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parent-Child hierarchical chunking: Small child chunks for high-precision
        similarity match + parent chunk context for rich LLM synthesis.
        """
        text = doc.get('content', '')
        sentences = self._split_into_sentences(text)
        parent_text = text
        chunks = []

        for idx, sentence in enumerate(sentences):
            chunks.append({
                'chunk_id': f"{doc['doc_id']}_child_{idx}",
                'doc_id': doc['doc_id'],
                'text': sentence,
                'parent_context': parent_text,
                'strategy': ChunkStrategy.HIERARCHICAL,
                'metadata': {
                    'title': doc.get('title', ''),
                    'is_child': True,
                    'parent_doc_id': doc['doc_id'],
                    'language': doc.get('language', 'en')
                }
            })
        return chunks

    def process_corpus(self, documents: List[Dict[str, Any]], strategy: str = "semantic") -> List[Dict[str, Any]]:
        all_chunks = []
        for doc in documents:
            if strategy == ChunkStrategy.FIXED_SIZE:
                all_chunks.extend(self.chunk_fixed_size(doc))
            elif strategy == ChunkStrategy.SLIDING_WINDOW:
                all_chunks.extend(self.chunk_sliding_window(doc))
            elif strategy == ChunkStrategy.METADATA_AWARE:
                all_chunks.extend(self.chunk_metadata_aware(doc))
            elif strategy == ChunkStrategy.HIERARCHICAL:
                all_chunks.extend(self.chunk_hierarchical(doc))
            else:  # Default to semantic
                all_chunks.extend(self.chunk_semantic(doc))
        return all_chunks
