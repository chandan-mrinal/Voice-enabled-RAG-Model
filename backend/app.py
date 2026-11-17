"""
Production FastAPI Application for Voice-Enabled RAG System
HH Goa 2026 Shortlisting Task 2
"""
import time
import os
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from .models import (
    RAGRequest, RAGResponse, LatencyBreakdown, GuardrailResult,
    RetrievedChunk, BenchmarkResult
)
from .dataset_loader import MSMARCOLoader
from .chunking import ChunkingEngine, ChunkStrategy
from .vector_engine import HybridVectorEngine
from .stt_service import STTService
from .guardrails import GuardrailEngine
from .harness import ModelHarness
from .latency_tracker import LatencyTracker
from .benchmarks import run_benchmark_suite, BENCHMARK_QUERIES

app = FastAPI(
    title="HH Goa 2026: Voice-Enabled RAG System",
    description="Sub-200ms Voice RAG with MSMARCO-XI, Sarvam/ElevenLabs, Vast Chunking, Guardrails & Latency Analytics",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core Service Singletons
loader = MSMARCOLoader()
chunking_engine = ChunkingEngine()
stt_service = STTService()
guardrail_engine = GuardrailEngine()
model_harness = ModelHarness()
latency_tracker = LatencyTracker()

# Pre-index vector engines for each chunking strategy for zero-cold-start sub-5ms retrieval
vector_engines: Dict[str, HybridVectorEngine] = {}

def initialize_vector_indices():
    raw_docs = loader.get_all_documents()
    strategies = [
        ChunkStrategy.SEMANTIC,
        ChunkStrategy.METADATA_AWARE,
        ChunkStrategy.SLIDING_WINDOW,
        ChunkStrategy.FIXED_SIZE,
        ChunkStrategy.HIERARCHICAL
    ]
    for strat in strategies:
        chunks = chunking_engine.process_corpus(raw_docs, strategy=strat)
        engine = HybridVectorEngine()
        engine.index_chunks(chunks)
        vector_engines[strat] = engine
    print(f"Initialized {len(vector_engines)} vector engines for all chunking strategies.")

initialize_vector_indices()

class RAGPipeline:
    def execute_query(
        self,
        query: str = None,
        audio_base64: str = None,
        stt_provider: str = "sarvam",
        chunking_strategy: str = "semantic",
        top_k: int = 3,
        enable_guardrails: bool = True,
        enable_hybrid_search: bool = True
    ) -> RAGResponse:
        t_total_start = time.perf_counter()
        
        # 1. STT Stage
        t_stt_start = time.perf_counter()
        if audio_base64:
            transcription, stt_latency = stt_service.transcribe(
                audio_base64=audio_base64,
                provider=stt_provider
            )
            final_query = transcription
        else:
            final_query = query if query else "What is photosynthesis?"
            stt_latency = (time.perf_counter() - t_stt_start) * 1000

        # 2. Input Guardrail Stage
        t_gr_start = time.perf_counter()
        if enable_guardrails:
            input_gr = guardrail_engine.validate_input(final_query)
            gr_input_latency = (time.perf_counter() - t_gr_start) * 1000
            
            if not input_gr.passed:
                total_latency = (time.perf_counter() - t_total_start) * 1000
                breakdown = LatencyBreakdown(
                    stt_ms=round(stt_latency, 2),
                    guardrail_input_ms=round(gr_input_latency, 2),
                    chunking_and_retrieval_ms=0.0,
                    harness_and_llm_ms=0.0,
                    guardrail_output_ms=0.0,
                    total_ms=round(total_latency, 2),
                    is_under_200ms=(total_latency <= 200.0)
                )
                latency_tracker.record(breakdown)
                return RAGResponse(
                    query=final_query,
                    transcription=final_query,
                    stt_provider=stt_provider,
                    chunking_strategy=chunking_strategy,
                    answer=input_gr.refusal_reason,
                    retrieved_chunks=[],
                    guardrail=input_gr,
                    latency=breakdown,
                    status="refused",
                    error_message=input_gr.refusal_reason
                )
        else:
            gr_input_latency = 0.0
            input_gr = GuardrailResult(passed=True)

        # 3. Chunking & Hybrid Retrieval Stage
        t_ret_start = time.perf_counter()
        engine = vector_engines.get(chunking_strategy, vector_engines[ChunkStrategy.SEMANTIC])
        retrieved_raw = engine.retrieve(final_query, top_k=top_k, hybrid=enable_hybrid_search)
        ret_latency = (time.perf_counter() - t_ret_start) * 1000

        formatted_chunks = [
            RetrievedChunk(
                chunk_id=c['chunk_id'],
                doc_id=c['doc_id'],
                text=c['text'],
                score=round(c.get('score', 0.0), 4),
                strategy_used=chunking_strategy,
                metadata=c.get('metadata', {})
            )
            for c in retrieved_raw
        ]

        # 4. Harness & Structured LLM Synthesis Stage
        t_llm_start = time.perf_counter()
        answer, harness_meta = model_harness.synthesize_answer(final_query, retrieved_raw)
        llm_latency = (time.perf_counter() - t_llm_start) * 1000

        # 5. Output Faithfulness / Hallucination Guardrail Stage
        t_gro_start = time.perf_counter()
        if enable_guardrails:
            output_gr = guardrail_engine.validate_output_grounding(final_query, answer, retrieved_raw)
            gr_output_latency = (time.perf_counter() - t_gro_start) * 1000

            if not output_gr.passed:
                total_latency = (time.perf_counter() - t_total_start) * 1000
                breakdown = LatencyBreakdown(
                    stt_ms=round(stt_latency, 2),
                    guardrail_input_ms=round(gr_input_latency, 2),
                    chunking_and_retrieval_ms=round(ret_latency, 2),
                    harness_and_llm_ms=round(llm_latency, 2),
                    guardrail_output_ms=round(gr_output_latency, 2),
                    total_ms=round(total_latency, 2),
                    is_under_200ms=(total_latency <= 200.0)
                )
                latency_tracker.record(breakdown)
                return RAGResponse(
                    query=final_query,
                    transcription=final_query,
                    stt_provider=stt_provider,
                    chunking_strategy=chunking_strategy,
                    answer=output_gr.refusal_reason,
                    retrieved_chunks=formatted_chunks,
                    guardrail=output_gr,
                    latency=breakdown,
                    status="refused",
                    error_message=output_gr.refusal_reason
                )
        else:
            gr_output_latency = 0.0
            output_gr = GuardrailResult(passed=True)

        total_latency = (time.perf_counter() - t_total_start) * 1000
        breakdown = LatencyBreakdown(
            stt_ms=round(stt_latency, 2),
            guardrail_input_ms=round(gr_input_latency, 2),
            chunking_and_retrieval_ms=round(ret_latency, 2),
            harness_and_llm_ms=round(llm_latency, 2),
            guardrail_output_ms=round(gr_output_latency, 2),
            total_ms=round(total_latency, 2),
            is_under_200ms=(total_latency <= 200.0)
        )
        latency_tracker.record(breakdown)

        return RAGResponse(
            query=final_query,
            transcription=final_query,
            stt_provider=stt_provider,
            chunking_strategy=chunking_strategy,
            answer=answer,
            retrieved_chunks=formatted_chunks,
            guardrail=output_gr,
            latency=breakdown,
            status="success",
            harness_metadata=harness_meta
        )

pipeline = RAGPipeline()

# REST Endpoints
@app.post("/api/query", response_model=RAGResponse)
async def query_rag(req: RAGRequest):
    return pipeline.execute_query(
        query=req.query,
        audio_base64=req.audio_base64,
        stt_provider=req.stt_provider,
        chunking_strategy=req.chunking_strategy,
        top_k=req.top_k,
        enable_guardrails=req.enable_guardrails,
        enable_hybrid_search=req.enable_hybrid_search
    )

@app.get("/api/benchmark", response_model=BenchmarkResult)
async def get_benchmarks():
    return run_benchmark_suite(pipeline)

@app.get("/api/latency-analytics")
async def get_latency_analytics():
    return latency_tracker.get_full_analytics()

@app.get("/api/chunking-comparison")
async def compare_chunking():
    raw_docs = loader.get_all_documents()
    res = {}
    for strat in [
        ChunkStrategy.SEMANTIC,
        ChunkStrategy.METADATA_AWARE,
        ChunkStrategy.SLIDING_WINDOW,
        ChunkStrategy.FIXED_SIZE,
        ChunkStrategy.HIERARCHICAL
    ]:
        chunks = chunking_engine.process_corpus(raw_docs, strategy=strat)
        lengths = [len(c['text'].split()) for c in chunks]
        res[strat] = {
            'total_chunks': len(chunks),
            'avg_words_per_chunk': round(sum(lengths) / max(1, len(lengths)), 1),
            'min_words': min(lengths) if lengths else 0,
            'max_words': max(lengths) if lengths else 0,
            'sample_chunk': chunks[0] if chunks else None
        }
    return res

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "indexed_documents": len(loader.get_all_documents()),
        "strategies": list(vector_engines.keys()),
        "latency_target": "<200ms"
    }

# Static UI mount
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
