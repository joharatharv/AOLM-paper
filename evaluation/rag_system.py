"""
RAG System for AOLM Math Tutor (evaluation/ copy)
Baseline Retrieval → Cross-Encoder Re-ranking → Graph Expansion
Identical to ../rag/rag_system.py — duplicated here so the evaluation/ folder
is self-contained for cluster runs.
"""

import json
import os
from typing import List, Dict, Tuple, Optional

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


EDGE_PRIORITY = [
    "prerequisites",
    "part_of",
    "demonstrates",
    "related_concepts",
    "related_to",
    "uses_technique",
    "used_in",
    "applies_to",
]
SKIP_EDGE_TYPES = {"detailed_version"}
GRAPH_SCORE_DECAY = 0.8


def load_chunks(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat_chunks = []
    for section in data:
        for chunk in section.get("chunks", []):
            chunk_id = chunk.get("chunk_id", "UNKNOWN")
            chunk_type = chunk.get("chunk_type", "unknown")
            meta = chunk.get("metadata", {})
            content = chunk.get("content", {})
            hints = chunk.get("retrieval_hints", {})

            text_parts = []
            topic = meta.get("topic", "")
            subtopic = meta.get("subtopic", "")
            if topic:    text_parts.append(f"Topic: {topic}")
            if subtopic: text_parts.append(f"Subtopic: {subtopic}")

            if "summary"     in content: text_parts.append(content["summary"])
            if "key_insight" in content: text_parts.append(f"Key insight: {content['key_insight']}")
            if "when_to_use" in content: text_parts.append(content["when_to_use"])

            if "formula_name" in content: text_parts.append(f"Formula: {content['formula_name']}")
            if "plain_text"   in content: text_parts.append(content["plain_text"])
            elif isinstance(content.get("key_formula"), dict):
                text_parts.append(content["key_formula"].get("plain", ""))
            if "conditions" in content: text_parts.append(f"Conditions: {content['conditions']}")
            if "quick_use"  in content: text_parts.append(content["quick_use"])

            if "problem"        in content: text_parts.append(f"Problem: {content['problem']}")
            if "solution_steps" in content and isinstance(content["solution_steps"], list):
                text_parts.append(" ".join(content["solution_steps"]))
            if "final_answer"   in content: text_parts.append(f"Answer: {content['final_answer']}")
            if "key_technique"  in content: text_parts.append(f"Technique: {content['key_technique']}")

            if "technique_name" in content: text_parts.append(f"Technique: {content['technique_name']}")
            if "step_by_step"   in content and isinstance(content["step_by_step"], list):
                text_parts.append(" ".join(content["step_by_step"]))
            if "recognition_pattern" in content: text_parts.append(content["recognition_pattern"])

            if "quick_example" in content and content["quick_example"]:
                text_parts.append(f"Example: {content['quick_example']}")

            keywords = hints.get("keywords", [])
            if keywords: text_parts.append(f"Keywords: {', '.join(keywords)}")

            embed_text = "\n".join(p for p in text_parts if p and not p.startswith("[GENERATED"))

            flat_chunks.append({
                "chunk_id":   chunk_id,
                "chunk_type": chunk_type,
                "topic":      topic,
                "difficulty": meta.get("difficulty", "unknown"),
                "embed_text": embed_text,
                "raw_content": content,
                "graph_links": chunk.get("graph_links", {}),
            })
    return flat_chunks


class RAGSystem:
    def __init__(
        self,
        chunks_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        collection_name: str = "aolm_math",
        persist_dir: str = "./chroma_db",
    ):
        print("[RAG] Loading chunks...")
        self.chunks = load_chunks(chunks_path)
        print(f"[RAG] Loaded {len(self.chunks)} chunks")

        self._build_graph()

        print(f"[RAG] Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        print(f"[RAG] Loading cross-encoder: {cross_encoder_model}")
        self.cross_encoder = CrossEncoder(cross_encoder_model)

        os.makedirs(persist_dir, exist_ok=True)
        print(f"[RAG] ChromaDB persist dir: {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        if self.collection.count() < len(self.chunks):
            self._index_chunks()

    def _build_graph(self):
        self.chunk_lookup = {c["chunk_id"]: c for c in self.chunks}
        self.chunk_graph: Dict[str, Dict[str, List[str]]] = {}

        for chunk in self.chunks:
            cid = chunk["chunk_id"]
            raw_links = chunk.get("graph_links", {})
            if not raw_links:
                continue
            neighbors: Dict[str, List[str]] = {}
            for edge_type, targets in raw_links.items():
                if edge_type in SKIP_EDGE_TYPES: continue
                target_list = targets if isinstance(targets, list) else [targets]
                real_targets = [
                    t for t in target_list
                    if isinstance(t, str) and not t.startswith("[GENERATED")
                ]
                if real_targets:
                    neighbors[edge_type] = real_targets
            if neighbors:
                self.chunk_graph[cid] = neighbors

        edge_count = sum(len(v) for adj in self.chunk_graph.values() for v in adj.values())
        print(f"[RAG] Graph: {len(self.chunk_graph)} nodes, {edge_count} edges")

    def _index_chunks(self):
        print("[RAG] Indexing chunks into ChromaDB...")
        ids   = [c["chunk_id"]   for c in self.chunks]
        texts = [c["embed_text"] for c in self.chunks]
        metas = [
            {"chunk_type": c["chunk_type"], "topic": c["topic"], "difficulty": c["difficulty"]}
            for c in self.chunks
        ]
        bs = 64
        for i in range(0, len(ids), bs):
            batch_ids   = ids[i : i + bs]
            batch_texts = texts[i : i + bs]
            batch_metas = metas[i : i + bs]
            batch_emb = self.embedder.encode(batch_texts).tolist()
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_emb,
                documents=batch_texts,
                metadatas=batch_metas,
            )
        print(f"[RAG] Indexed {self.collection.count()} chunks")

    def baseline_retrieval(self, query: str, top_k: int = 10) -> List[Dict]:
        q_emb = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "chunk_id": results["ids"][0][i],
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return retrieved

    def rerank(self, query: str, candidates: List[Dict], top_n: int = 3) -> List[Dict]:
        if not candidates: return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_n]

    def graph_expand(self, seeds: List[Dict], max_neighbors: int = 2) -> List[Dict]:
        seen_ids = {c["chunk_id"] for c in seeds}
        expansion: List[Dict] = []
        for seed in seeds:
            if len(expansion) >= max_neighbors: break
            cid = seed["chunk_id"]
            adjacency = self.chunk_graph.get(cid, {})
            seed_score = seed.get("rerank_score", 0.0)
            for edge_type in EDGE_PRIORITY:
                if len(expansion) >= max_neighbors: break
                for neighbor_id in adjacency.get(edge_type, []):
                    if neighbor_id in seen_ids: continue
                    if neighbor_id not in self.chunk_lookup: continue
                    seen_ids.add(neighbor_id)
                    nb = self.chunk_lookup[neighbor_id]
                    expansion.append({
                        "chunk_id": neighbor_id,
                        "text":     nb["embed_text"],
                        "metadata": {
                            "chunk_type": nb["chunk_type"],
                            "topic":      nb["topic"],
                            "difficulty": nb["difficulty"],
                        },
                        "rerank_score": seed_score * GRAPH_SCORE_DECAY,
                        "graph_source": True,
                        "graph_edge":   edge_type,
                        "graph_seed":   cid,
                    })
                    if len(expansion) >= max_neighbors: break
        return expansion

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 3,
        min_score: float = 0.5,
        use_graph: bool = True,
        max_graph_neighbors: int = 2,
    ) -> List[Dict]:
        candidates = self.baseline_retrieval(query, top_k=top_k)
        seeds = self.rerank(query, candidates, top_n=top_n)
        if not seeds: return []
        best_score = seeds[0].get("rerank_score", float("-inf"))
        if best_score < min_score:
            print(f"[RAG] Score gate triggered (best={best_score:.3f} < {min_score}); no context.")
            return []
        filtered = [s for s in seeds if s.get("rerank_score", float("-inf")) >= min_score]
        if not filtered: return []
        if use_graph:
            expanded = self.graph_expand(filtered, max_neighbors=max_graph_neighbors)
            return filtered + expanded
        return filtered

    def format_context(self, chunks: List[Dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            if c.get("graph_source"):
                src = f"graph-expanded via '{c['graph_edge']}' from {c['graph_seed']}"
            else:
                src = "direct retrieval"
            parts.append(
                f"--- Reference {i} [{c['chunk_id']}] "
                f"(type: {c['metadata']['chunk_type']}, "
                f"topic: {c['metadata']['topic']}, source: {src}) ---\n"
                f"{c['text']}"
            )
        return "\n\n".join(parts)
