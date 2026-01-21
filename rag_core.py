import os, re, json, hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
import requests
from openai import OpenAI


# =========================
# 0) Models / Keys
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY (set env var).")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY (set env var).")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

ROUTER_MODEL = os.getenv("DEEPSEEK_ROUTER_MODEL", "deepseek-chat")
RERANK_MODEL = os.getenv("DEEPSEEK_RERANK_MODEL", "deepseek-chat")
ANSWER_MODEL = os.getenv("DEEPSEEK_ANSWER_MODEL", "deepseek-chat")

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")  # 3072 dim


# =========================
# 1) DeepSeek chat
# =========================
def ds_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 900) -> str:
    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def ds_chat_stream(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 900,
):
    """
    DeepSeek / OpenAI-compatible streaming chat
    yield: 생성되는 텍스트 조각들
    """
    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    with requests.post(
        url,
        headers=headers,
        data=json.dumps(payload),
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                    delta = j["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]
                except Exception:
                    continue

# =========================
# 2) Embeddings (OpenAI)
# =========================
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns: np.ndarray [n, dim] float32, normalized for cosine similarity.
    """
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


# =========================
# 3) Data model
# =========================
@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


def safe_json_load(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return {}
        return json.loads(m.group(0))


def format_citation(md: Dict[str, Any]) -> str:
    src = md.get("source", "unknown")
    if md.get("page") is not None:
        return f"{src} p.{md.get('page')}"
    if md.get("row") is not None:
        return f"{src} row {md.get('row')}"
    if md.get("date"):
        return f"{src} ({md.get('date')})"
    return src


# =========================
# 4) VectorStore (load-only)
# =========================
class VectorStore:
    def __init__(self, index: faiss.Index, chunks: List[Chunk]):
        self.index = index
        self.chunks = chunks

    def search(self, query: str, top_k: int = 50) -> List[Tuple[Chunk, float]]:
        qv = embed_texts([query])
        scores, idxs = self.index.search(qv, top_k)
        out: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append((self.chunks[idx], float(score)))
        return out


def _load_chunks_jsonl(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            chunks.append(Chunk(
                chunk_id=j["chunk_id"],
                text=j["text"],
                metadata=j.get("metadata", {})
            ))
    return chunks


def _load_chunks_pickle(path: str) -> List[Chunk]:
    import pickle
    import sys

    # 🔥 핵심: pickle에 저장된 main.Chunk를 현재 Chunk로 매핑
    sys.modules["main"] = sys.modules[__name__]
    with open(path, "rb") as f:
        raw = pickle.load(f)

    # raw가 이미 Chunk list인 경우
    if raw and isinstance(raw[0], Chunk):
        return raw

    # dict list인 경우
    chunks: List[Chunk] = []
    for j in raw:
        chunks.append(Chunk(chunk_id=j["chunk_id"], text=j["text"], metadata=j.get("metadata", {})))
    return chunks


def load_vectorstore(vs_dir: str) -> VectorStore:
    """
    vs_dir 아래에 아래 파일들 중 하나 조합이 있어야 함:
      - index.faiss  (필수)
      - chunks.jsonl 또는 chunks.pkl (필수)

    ※ 인덱스와 chunks 순서가 정확히 매칭되어야 함.
    """
    index_path = os.path.join(vs_dir, "faiss.index")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    chunks_jsonl = os.path.join(vs_dir, "chunks.jsonl")
    chunks_pkl = os.path.join(vs_dir, "chunks.pkl")

    if os.path.exists(chunks_jsonl):
        chunks = _load_chunks_jsonl(chunks_jsonl)
    elif os.path.exists(chunks_pkl):
        chunks = _load_chunks_pickle(chunks_pkl)
    else:
        raise FileNotFoundError(f"chunks file not found. Need chunks.jsonl or chunks.pkl under: {vs_dir}")

    index = faiss.read_index(index_path)
    return VectorStore(index=index, chunks=chunks)


# =========================
# 5) Router / Retrieval / Rerank / Answer (원래 구조 유지)
# =========================
def route_query(question: str, catalog_text: str, default_sources: List[str]) -> Dict[str, Any]:
    system = (
        "너는 RAG 라우터다. 사용자 질문을 보고 어떤 파일(source)에서 검색해야 하는지 선택해야 한다.\n"
        "아래 카탈로그의 source 이름 중에서만 고른다.\n"
        "애매하면 sources를 2~4개로 넓힌다.\n"
        "시간/시기 조건(예: 1960년대)이 있으면 filters에 넣는다.\n"
        "집계(몇 번/가장 많이/언제 가장 활발) 질문이면 need_counting=true.\n"
        "출력은 JSON 하나만.\n\n"
        f"카탈로그:\n{catalog_text}\n"
    )
    user = (
        f"질문: {question}\n\n"
        "JSON 스키마:\n"
        "{\n"
        '  "sources": ["..."],\n'
        '  "query_rewrite": "검색에 적합한 키워드 중심 쿼리(한/영 혼합 가능)",\n'
        '  "filters": {"year_from": null, "year_to": null, "decade": null},\n'
        '  "need_counting": false,\n'
        '  "confidence": 0.0\n'
        "}\n"
    )
    out = ds_chat(ROUTER_MODEL, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                  temperature=0.0, max_tokens=400)
    j = safe_json_load(out) or {}

    if not j.get("sources"):
        j["sources"] = default_sources[:]  # fallback defaults
        j["confidence"] = float(j.get("confidence", 0.0)) * 0.5

    j["query_rewrite"] = j.get("query_rewrite") or question
    j["filters"] = j.get("filters") or {}
    j["need_counting"] = bool(j.get("need_counting", False))
    j["confidence"] = float(j.get("confidence", 0.0))

    # confidence 낮으면 기본적으로 넓힘
    if j["confidence"] < 0.6:
        j["sources"] = list(dict.fromkeys(j["sources"] + default_sources))[:4]

    return j


def apply_filters(results: List[Tuple[Chunk, float]], sources: List[str], filters: Dict[str, Any]) -> List[Tuple[Chunk, float]]:
    src_set = set(sources)
    year_from = filters.get("year_from")
    year_to = filters.get("year_to")
    decade = filters.get("decade")

    def ok(c: Chunk) -> bool:
        md = c.metadata
        if md.get("source") not in src_set:
            return False
        y = md.get("year")
        if isinstance(year_from, int) and isinstance(y, int) and y < year_from:
            return False
        if isinstance(year_to, int) and isinstance(y, int) and y > year_to:
            return False
        if decade and md.get("decade") and md.get("decade") != decade:
            return False
        return True

    return [(c, s) for (c, s) in results if ok(c)]


def retrieve(vs: VectorStore, query_rewrite: str, sources: List[str], filters: Dict[str, Any],
             initial_k: int = 80, final_k: int = 40) -> List[Tuple[Chunk, float]]:
    raw = vs.search(query_rewrite, top_k=initial_k)
    filtered = apply_filters(raw, sources, filters)

    if len(filtered) < 10:
        filtered = apply_filters(raw, sources, {})  # relax filters
    return filtered[:final_k]


def rerank_llm(question: str, candidates: List[Tuple[Chunk, float]], top_n: int = 10) -> List[Chunk]:
    items = []
    for i, (c, score) in enumerate(candidates):
        md = c.metadata
        items.append({
            "i": i,
            "score": round(score, 4),
            "source": md.get("source"),
            "page": md.get("page"),
            "row": md.get("row"),
            "date": md.get("date"),
            "year": md.get("year"),
            "snippet": c.text[:450],
        })

    system = (
        "너는 RAG 근거 선별기다.\n"
        "질문에 직접 답이 되는 근거만 고른다.\n"
        "시간 조건이 있으면 그 조건을 만족하는 근거를 우선한다.\n"
        "출력은 JSON 하나만: {\"picked\": [indices...]}\n"
    )
    user = json.dumps({"question": question, "candidates": items, "top_n": top_n}, ensure_ascii=False)
    out = ds_chat(RERANK_MODEL, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                  temperature=0.0, max_tokens=300)

    j = safe_json_load(out) or {}
    picked = j.get("picked", [])
    if not isinstance(picked, list) or not picked:
        picked = list(range(min(top_n, len(candidates))))

    picked = [int(x) for x in picked if str(x).isdigit()]
    picked = [x for x in picked if 0 <= x < len(candidates)][:top_n]
    if not picked:
        picked = list(range(min(top_n, len(candidates))))

    return [candidates[i][0] for i in picked]



def answer_stream(question: str, evidence: List[Chunk], mode: str = "grounded", history: Optional[List[Dict[str,str]]] = None):
    ev_lines = []
    for i, c in enumerate(evidence, 1):
        ev_lines.append(f"[근거 {i}] ({format_citation(c.metadata)}) {c.text}")

    if mode == "grounded":
        system = (
            "너는 '백남준' 지식 기반 도슨트 챗봇이다.\n"
            "사용자의 질문에 자연스럽고 대화하듯 답해라.\n"
            "반드시 제공된 '근거' 안에서만 답해라.\n"
            "답변은 번호/섹션으로 나누지 말고 자연스럽게 서술해라.\n"
            "이상한 기호(** 같은 것) 넣지 말고 자연스럽게 써라.\n"
            "출처 목록은 작성하지 말고, 답변 내용만 작성해라.\n"
        )
        user = "질문:\n" + question + "\n\n근거:\n" + "\n".join(ev_lines)
    else:
        system = (
            "너는 '백남준' 지식 기반 도슨트 챗봇이다.\n"
            "이번에는 제공된 RAG 근거가 없거나 매우 부족하다.\n"
            "그래도 질문에 대해 가지고 있는 일반 지식과 추론으로 최대한 유용하게 답해라.\n"
            "이상한 기호(** 같은 것) 넣지 말고 자연스럽게 써라.\n"
            "출처 목록은 작성하지 말고, 답변 내용만 작성해라.\n"
        )
        user = "질문:\n" + question
    messages = [{"role":"system","content":system}]
    if history:
        for m in history:
            if m.get("role") in ("user","assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role":"user","content":user})

    return ds_chat_stream(
        ANSWER_MODEL,
        messages,
        temperature=0.2,
        max_tokens=900
    )

def rag_qa(
    vs: VectorStore,
    question: str,
    catalog_text: str,
    all_sources: List[str],
    default_sources: List[str],
    history: Optional[List[Dict[str, str]]] = None,  # (현재는 answer_stream에서만 사용)
    use_rerank: bool = True,
    top_n_evidence: int = 8,
    score_threshold: float = 0.22,
) -> Dict[str, Any]:
    """
    ✅ 여기서는 '답변을 생성하지 않는다'
    - 라우팅/검색/리랭크까지 해서 evidence만 준비하고,
    - 스트리밍 답변은 app.py에서 answer_stream으로 생성한다.
    """

    route = route_query(question, catalog_text=catalog_text, default_sources=default_sources)

    sources = route["sources"]
    query_rewrite = route["query_rewrite"]
    filters = route["filters"]

    candidates = retrieve(vs, query_rewrite, sources, filters, initial_k=80, final_k=40)

    # 후보가 너무 적으면 소스 확장(구조 유지)
    if len(candidates) < 5:
        candidates = retrieve(vs, query_rewrite, all_sources, {}, initial_k=120, final_k=50)

    if use_rerank:
        evidence = rerank_llm(question, candidates, top_n=top_n_evidence)
    else:
        evidence = [c for (c, _) in candidates[:top_n_evidence]]

    top_score = candidates[0][1] if candidates else 0.0
    evidence_insufficient = (len(evidence) < 2) or (top_score < score_threshold)

    mode = "fallback" if evidence_insufficient else "grounded"

    cite_lines = [f"[{i}] {format_citation(c.metadata)}" for i, c in enumerate(evidence, 1)]

    return {
        "question": question,
        "route": route,
        "top_score": top_score,
        "evidence": [{"cite": format_citation(c.metadata), "text": c.text[:240]} for c in evidence],
        "evidence_chunks": evidence,   # ✅ 스트리밍에 그대로 넣을 Chunk들
        "cite_lines": cite_lines,      # ✅ app.py에서 붙일 출처 라인
        "mode": mode,
    }
