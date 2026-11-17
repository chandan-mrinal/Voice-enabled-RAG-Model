"""
Pydantic Data Models for Structured I/O, Harness, Guardrails, and Latency Tracking
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class LatencyBreakdown(BaseModel):
    stt_ms: float = 0.0
    guardrail_input_ms: float = 0.0
    chunking_and_retrieval_ms: float = 0.0
    harness_and_llm_ms: float = 0.0
    guardrail_output_ms: float = 0.0
    total_ms: float = 0.0
    is_under_200ms: bool = True

class GuardrailResult(BaseModel):
    passed: bool = True
    input_safety_passed: bool = True
    domain_relevance_passed: bool = True
    grounding_passed: bool = True
    risk_category: Optional[str] = None
    refusal_reason: Optional[str] = None
    confidence_score: float = 1.0
    grounding_score: float = 1.0

class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    strategy_used: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RAGRequest(BaseModel):
    query: Optional[str] = None
    audio_base64: Optional[str] = None
    stt_provider: Literal["sarvam", "elevenlabs", "mock_fast"] = "sarvam"
    chunking_strategy: Literal["semantic", "metadata_aware", "sliding_window", "hierarchical", "fixed_size"] = "semantic"
    top_k: int = 3
    enable_guardrails: bool = True
    enable_hybrid_search: bool = True

class RAGResponse(BaseModel):
    query: str
    transcription: str
    stt_provider: str
    chunking_strategy: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    guardrail: GuardrailResult
    latency: LatencyBreakdown
    status: Literal["success", "refused", "error"] = "success"
    error_message: Optional[str] = None
    harness_metadata: Dict[str, Any] = Field(default_factory=dict)

class BenchmarkResult(BaseModel):
    query_count: int
    p50_total_ms: float
    p70_total_ms: float
    p100_total_ms: float
    p50_retrieval_ms: float
    p70_retrieval_ms: float
    p100_retrieval_ms: float
    p50_stt_ms: float
    p70_stt_ms: float
    p100_stt_ms: float
    p50_llm_ms: float
    p70_llm_ms: float
    p100_llm_ms: float
    success_rate: float
    guardrail_accuracy: float
    under_200ms_percentage: float
    runs: List[Dict[str, Any]]
