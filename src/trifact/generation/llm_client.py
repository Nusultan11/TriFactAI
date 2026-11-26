"""Grounded LLM generation clients (llama.cpp / vLLM OpenAI-compatible, HF Transformers+PEFT, RAG pipeline).

Implements ``trifact.core.interfaces.Generator`` (ADR-0012, ADR-0023).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from trifact.core.interfaces import Generator, Reranker, Retriever
from trifact.core.schemas import Lang, RetrievalScope, ScoredChunk
from trifact.generation.prompt import (
    REFUSAL_PHRASES,
    GroundedPrompt,
    build_grounded_prompt,
    detect_repetition_loop,
    is_refusal,
    map_and_validate_citations,
    strip_thinking_tags,
)

log = logging.getLogger("trifact.generation.llm_client")

DEFAULT_STOP_SEQUENCES: tuple[str, ...] = ("<|im_end|>", "<|endoftext|>", "\nQuestion:", "\nEvidence:")


@dataclass
class GenerationResult:
    """Detailed generation output with latency, token throughput, and attribution diagnostics."""

    answer: str
    raw_answer: str
    answer_lang: str
    refused: bool
    valid_chunk_ids: list[str]
    invalid_citations: list[str]
    had_think_tags: bool
    repetition_loop: bool
    ttft_ms: float
    total_ms: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_s: float
    model_version: str
    prompt_info: GroundedPrompt | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _finalize_generation(
    raw_text: str,
    prompt_info: GroundedPrompt,
    *,
    ttft_ms: float,
    total_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    model_version: str,
    extra: dict[str, Any] | None = None,
) -> GenerationResult:
    cleaned, had_think = strip_thinking_tags(raw_text)
    mapped, valid_ids, invalid_ids = map_and_validate_citations(cleaned, prompt_info.id_to_chunk_id)
    refused = is_refusal(mapped, prompt_info.answer_lang)
    rep_loop = detect_repetition_loop(mapped)
    decode_s = max((total_ms - ttft_ms) / 1000.0, total_ms / 1000.0, 1e-6)
    tok_s = completion_tokens / decode_s if completion_tokens > 0 else 0.0
    return GenerationResult(
        answer=mapped,
        raw_answer=raw_text,
        answer_lang=prompt_info.answer_lang,
        refused=refused,
        valid_chunk_ids=valid_ids,
        invalid_citations=invalid_ids,
        had_think_tags=had_think,
        repetition_loop=rep_loop,
        ttft_ms=round(ttft_ms, 2),
        total_ms=round(total_ms, 2),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_s=round(tok_s, 2),
        model_version=model_version,
        prompt_info=prompt_info,
        extra=extra or {},
    )


@dataclass
class OpenAICompatibleLLMClient:
    """OpenAI-compatible client for ``llama.cpp`` server and ``vLLM`` (ADR-0012).

    Satisfies ``trifact.core.interfaces.Generator``.
    """

    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "Qwen/Qwen3.5-2B"
    api_key: str = "not-needed"
    version: str = ""
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    timeout_s: float = 120.0
    thinking: bool = False
    stream: bool = True
    use_short_ids: bool = True
    spotlight: bool = True
    max_chunks: int = 8
    max_context_tokens: int = 2400
    stop_sequences: tuple[str, ...] = DEFAULT_STOP_SEQUENCES
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.version:
            self.version = f"{self.model}@openai-compat"

    def healthcheck(self) -> bool:
        """Check whether the upstream LLM server responds on ``/health`` or ``/v1/models``."""
        root_url = self.base_url.removesuffix("/v1")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{root_url}/health", headers=headers)
                if r.status_code == 200:
                    return True
                r2 = client.get(f"{self.base_url}/models", headers=headers)
                return r2.status_code == 200
        except httpx.HTTPError:
            return False

    def _build_payload(self, prompt_info: GroundedPrompt, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": prompt_info.messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": stream,
            "stop": list(self.stop_sequences),
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        payload.update(self.extra_body)
        return payload

    def generate_detailed(
        self,
        question: str,
        context: Sequence[ScoredChunk],
        answer_lang: Lang | str,
        *,
        client: httpx.Client | None = None,
    ) -> GenerationResult:
        """Run grounded generation and return full timing/attribution metadata."""
        prompt_info = build_grounded_prompt(
            question,
            context,
            answer_lang,
            max_chunks=self.max_chunks,
            max_context_tokens=self.max_context_tokens,
            use_short_ids=self.use_short_ids,
            spotlight=self.spotlight,
        )
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = self._build_payload(prompt_info, stream=self.stream)

        own_client = client is None
        http = client or httpx.Client(timeout=self.timeout_s)
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        chunks_out: list[str] = []
        prompt_tokens = prompt_info.estimated_context_tokens
        completion_tokens = 0

        try:
            if self.stream:
                with http.stream("POST", url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if not raw_line or not raw_line.startswith("data:"):
                            continue
                        data_str = raw_line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            evt = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        usage = evt.get("usage")
                        if isinstance(usage, dict):
                            prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                            completion_tokens = int(usage.get("completion_tokens", completion_tokens))
                        choices = evt.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            piece = delta.get("content") or ""
                            if piece:
                                if ttft_ms is None:
                                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                                chunks_out.append(piece)
            else:
                resp = http.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()
                ttft_ms = (time.perf_counter() - t0) * 1000.0
                usage = body.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                completion_tokens = int(usage.get("completion_tokens", 0))
                choices = body.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    chunks_out.append(str(msg.get("content") or ""))
        finally:
            if own_client:
                http.close()

        total_ms = (time.perf_counter() - t0) * 1000.0
        raw_text = "".join(chunks_out)
        if completion_tokens == 0 and raw_text:
            completion_tokens = max(1, len(raw_text.split()))
        return _finalize_generation(
            raw_text,
            prompt_info,
            ttft_ms=ttft_ms if ttft_ms is not None else total_ms,
            total_ms=total_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_version=self.version,
        )

    def generate(self, question: str, context: Sequence[ScoredChunk], answer_lang: Lang) -> str:
        """Generate grounded cited answer string (``trifact.core.interfaces.Generator``)."""
        return self.generate_detailed(question, context, answer_lang).answer


@dataclass
class TransformersGenerator:
    """Hugging Face ``transformers`` (+ optional PEFT LoRA adapter) generator for cloud GPU eval.

    Satisfies ``trifact.core.interfaces.Generator``.
    """

    model_name_or_path: str = "Qwen/Qwen3.5-2B"
    adapter_path: str | None = None
    version: str = ""
    device: str | None = None
    torch_dtype: str = "auto"  # auto | bf16 | fp16 | fp32
    max_new_tokens: int = 256
    temperature: float = 0.0
    thinking: bool = False
    use_short_ids: bool = True
    spotlight: bool = True
    max_chunks: int = 8
    max_context_tokens: int = 2400

    def __post_init__(self) -> None:
        if not self.version:
            suffix = "+lora" if self.adapter_path else ""
            self.version = f"{self.model_name_or_path}{suffix}"
        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        if self.torch_dtype in dtype_map:
            dtype = dtype_map[self.torch_dtype]
        elif self.device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        log.info(
            "Loading TransformersGenerator %s (adapter=%s, device=%s, dtype=%s)",
            self.model_name_or_path,
            self.adapter_path,
            self.device,
            dtype,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype=dtype,
            device_map="auto" if self.device.startswith("cuda") else None,
        )
        if not self.device.startswith("cuda"):
            model.to(self.device)
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._model = model

    def _apply_template(self, messages: list[dict[str, str]]) -> str:
        tok = self._tokenizer
        try:
            return str(
                tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.thinking,
                )
            )
        except TypeError:
            return str(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    def generate_detailed(
        self,
        question: str,
        context: Sequence[ScoredChunk],
        answer_lang: Lang | str,
    ) -> GenerationResult:
        import torch

        self._ensure_loaded()
        tok = self._tokenizer
        model = self._model

        def _count(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))

        prompt_info = build_grounded_prompt(
            question,
            context,
            answer_lang,
            max_chunks=self.max_chunks,
            max_context_tokens=self.max_context_tokens,
            use_short_ids=self.use_short_ids,
            spotlight=self.spotlight,
            token_counter=_count,
        )
        rendered = self._apply_template(prompt_info.messages)
        inputs = tok(rendered, return_tensors="pt")
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
            "pad_token_id": tok.eos_token_id,
        }
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature

        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)
        total_ms = (time.perf_counter() - t0) * 1000.0

        gen_tokens = out_ids[0][prompt_tokens:]
        completion_tokens = int(gen_tokens.shape[0])
        raw_text = tok.decode(gen_tokens, skip_special_tokens=True)

        # Estimate TTFT as single-token fraction when non-streaming generate() is used
        ttft_ms = total_ms / max(completion_tokens, 1)
        return _finalize_generation(
            raw_text,
            prompt_info,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_version=self.version,
        )

    def generate(self, question: str, context: Sequence[ScoredChunk], answer_lang: Lang) -> str:
        return self.generate_detailed(question, context, answer_lang).answer


@dataclass
class ExtractiveBaselineGenerator:
    """Fast deterministic extractive baseline (no model downloads) for unit tests and control checks.

    Selects the highest-overlap sentence from the top-scored evidence chunk and cites its ID,
    or returns the canonical refusal phrase when overlap is below ``min_overlap``.
    """

    version: str = "extractive-baseline@v1"
    min_overlap: float = 0.15
    use_short_ids: bool = True
    spotlight: bool = True
    max_chunks: int = 8
    max_context_tokens: int = 2400

    def generate_detailed(
        self,
        question: str,
        context: Sequence[ScoredChunk],
        answer_lang: Lang | str,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        prompt_info = build_grounded_prompt(
            question,
            context,
            answer_lang,
            max_chunks=self.max_chunks,
            max_context_tokens=self.max_context_tokens,
            use_short_ids=self.use_short_ids,
            spotlight=self.spotlight,
        )
        q_toks = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 2}
        best_sent = ""
        best_id = ""
        best_score = 0.0

        for sc in prompt_info.selected_chunks:
            disp_id = prompt_info.chunk_id_to_id[sc.chunk.chunk_id]
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", sc.chunk.text) if s.strip()]
            for sent in sents:
                s_toks = {w for w in re.findall(r"\w+", sent.lower()) if len(w) > 2}
                if not q_toks or not s_toks:
                    continue
                overlap = len(q_toks & s_toks) / len(q_toks)
                if overlap > best_score:
                    best_score = overlap
                    best_sent = sent
                    best_id = disp_id

        if best_score < self.min_overlap or not best_sent:
            raw_text = REFUSAL_PHRASES[prompt_info.answer_lang]
        else:
            raw_text = f"{best_sent.rstrip('.')} [{best_id}]."

        total_ms = (time.perf_counter() - t0) * 1000.0
        return _finalize_generation(
            raw_text,
            prompt_info,
            ttft_ms=total_ms,
            total_ms=total_ms,
            prompt_tokens=prompt_info.estimated_context_tokens,
            completion_tokens=max(1, len(raw_text.split())),
            model_version=self.version,
        )

    def generate(self, question: str, context: Sequence[ScoredChunk], answer_lang: Lang) -> str:
        return self.generate_detailed(question, context, answer_lang).answer


@dataclass
class RAGTrace:
    """End-to-end retrieval -> rerank -> generation trace for Stage 7/9 diagnostics."""

    question: str
    answer_lang: str
    retrieved_ids: list[str]
    reranked_ids: list[str]
    prompt_chunk_ids: list[str]
    generation: GenerationResult
    retrieve_ms: float
    rerank_ms: float


@dataclass
class GroundedRAGPipeline:
    """Integrates Stage 6 ``Retriever`` + ``Reranker`` with Stage 7 ``Generator``."""

    retriever: Retriever
    generator: Generator
    reranker: Reranker | None = None
    candidates_k: int = 50
    context_k: int = 8
    early_refusal_threshold: float | None = None

    def answer_with_trace(
        self,
        question: str,
        answer_lang: Lang,
        *,
        scope: RetrievalScope = RetrievalScope.DOCUMENTS,
        workspace_id: str | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> RAGTrace:
        t0 = time.perf_counter()
        retrieved = self.retriever.search(
            query=question,
            top_k=self.candidates_k,
            scope=scope,
            workspace_id=workspace_id,
            document_ids=document_ids,
        )
        retrieve_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        if self.reranker is not None and retrieved:
            reranked = self.reranker.rerank(question, retrieved, top_k=self.context_k)
        else:
            reranked = list(retrieved[: self.context_k])
        rerank_ms = (time.perf_counter() - t1) * 1000.0

        # Optional calibrated early refusal (trifact-rag-pipeline)
        if self.early_refusal_threshold is not None:
            top_score = max((c.score for c in reranked), default=-1e9)
            if top_score < self.early_refusal_threshold:
                lang_s = answer_lang.value
                refusal_text = REFUSAL_PHRASES[lang_s]
                gen_res = GenerationResult(
                    answer=refusal_text,
                    raw_answer=refusal_text,
                    answer_lang=lang_s,
                    refused=True,
                    valid_chunk_ids=[],
                    invalid_citations=[],
                    had_think_tags=False,
                    repetition_loop=False,
                    ttft_ms=0.0,
                    total_ms=0.0,
                    prompt_tokens=0,
                    completion_tokens=len(refusal_text.split()),
                    tokens_per_s=0.0,
                    model_version=self.generator.version,
                    extra={"early_refusal": True, "top_rerank_score": top_score},
                )
                return RAGTrace(
                    question=question,
                    answer_lang=lang_s,
                    retrieved_ids=[c.chunk.chunk_id for c in retrieved],
                    reranked_ids=[c.chunk.chunk_id for c in reranked],
                    prompt_chunk_ids=[],
                    generation=gen_res,
                    retrieve_ms=round(retrieve_ms, 2),
                    rerank_ms=round(rerank_ms, 2),
                )

        if hasattr(self.generator, "generate_detailed"):
            gen_fn: Callable[..., GenerationResult] = self.generator.generate_detailed
            gen_res = gen_fn(question, reranked, answer_lang)
        else:
            t2 = time.perf_counter()
            ans = self.generator.generate(question, reranked, answer_lang)
            dt = (time.perf_counter() - t2) * 1000.0
            gen_res = GenerationResult(
                answer=ans,
                raw_answer=ans,
                answer_lang=answer_lang.value,
                refused=is_refusal(ans, answer_lang),
                valid_chunk_ids=[],
                invalid_citations=[],
                had_think_tags=False,
                repetition_loop=detect_repetition_loop(ans),
                ttft_ms=round(dt, 2),
                total_ms=round(dt, 2),
                prompt_tokens=0,
                completion_tokens=max(1, len(ans.split())),
                tokens_per_s=0.0,
                model_version=self.generator.version,
            )

        prompt_ids = (
            [c.chunk.chunk_id for c in gen_res.prompt_info.selected_chunks]
            if gen_res.prompt_info is not None
            else [c.chunk.chunk_id for c in reranked]
        )
        return RAGTrace(
            question=question,
            answer_lang=answer_lang.value,
            retrieved_ids=[c.chunk.chunk_id for c in retrieved],
            reranked_ids=[c.chunk.chunk_id for c in reranked],
            prompt_chunk_ids=prompt_ids,
            generation=gen_res,
            retrieve_ms=round(retrieve_ms, 2),
            rerank_ms=round(rerank_ms, 2),
        )
