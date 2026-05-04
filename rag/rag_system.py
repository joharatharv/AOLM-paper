"""
RAG System for CurioChain Math Tutor
Baseline Retrieval → Cross-Encoder Re-ranking → Graph Expansion → LLM Generation
"""

import json
import os
from typing import List, Dict, Tuple, Optional

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

# Edge types ordered by retrieval priority for graph expansion.
# prerequisites / part_of / demonstrates pull in foundational context;
# related_concepts / related_to / uses_technique add siblings;
# used_in / applies_to pull in downstream examples (lowest priority).
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

# detailed_version links point to *_FULL chunks that aren't in the KB — skip them.
SKIP_EDGE_TYPES = {"detailed_version"}

# Score decay applied to graph-expanded neighbors relative to their seed chunk.
GRAPH_SCORE_DECAY = 0.8


# ─────────────────────────────────────────────────────────────────
# 1. CHUNK LOADER — Flattens concepts.json into embeddable strings
# ─────────────────────────────────────────────────────────────────

def load_chunks(json_path: str) -> List[Dict]:
    """Load and flatten all chunks from concepts.json into a list of dicts."""
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

            # Build a single embeddable text from content fields
            text_parts = []

            # Add topic context
            topic = meta.get("topic", "")
            subtopic = meta.get("subtopic", "")
            if topic:
                text_parts.append(f"Topic: {topic}")
            if subtopic:
                text_parts.append(f"Subtopic: {subtopic}")

            # Core concept fields
            if "summary" in content:
                text_parts.append(content["summary"])
            if "key_insight" in content:
                text_parts.append(f"Key insight: {content['key_insight']}")
            if "when_to_use" in content:
                text_parts.append(content["when_to_use"])

            # Formula fields
            if "formula_name" in content:
                text_parts.append(f"Formula: {content['formula_name']}")
            if "plain_text" in content:
                text_parts.append(content["plain_text"])
            elif isinstance(content.get("key_formula"), dict):
                text_parts.append(content["key_formula"].get("plain", ""))
            if "conditions" in content:
                text_parts.append(f"Conditions: {content['conditions']}")
            if "quick_use" in content:
                text_parts.append(content["quick_use"])

            # Worked example fields
            if "problem" in content:
                text_parts.append(f"Problem: {content['problem']}")
            if "solution_steps" in content and isinstance(content["solution_steps"], list):
                text_parts.append(" ".join(content["solution_steps"]))
            if "final_answer" in content:
                text_parts.append(f"Answer: {content['final_answer']}")
            if "key_technique" in content:
                text_parts.append(f"Technique: {content['key_technique']}")

            # Technique fields
            if "technique_name" in content:
                text_parts.append(f"Technique: {content['technique_name']}")
            if "step_by_step" in content and isinstance(content["step_by_step"], list):
                text_parts.append(" ".join(content["step_by_step"]))
            if "recognition_pattern" in content:
                text_parts.append(content["recognition_pattern"])

            # Quick example
            if "quick_example" in content and content["quick_example"]:
                text_parts.append(f"Example: {content['quick_example']}")

            # Retrieval hint keywords (boost embedding quality)
            keywords = hints.get("keywords", [])
            if keywords:
                text_parts.append(f"Keywords: {', '.join(keywords)}")

            embed_text = "\n".join(p for p in text_parts if p and not p.startswith("[GENERATED"))

            flat_chunks.append({
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "topic": topic,
                "difficulty": meta.get("difficulty", "unknown"),
                "embed_text": embed_text,
                "raw_content": content,
                # Preserve graph links for graph expansion (raw from source)
                "graph_links": chunk.get("graph_links", {}),
            })

    return flat_chunks


# ─────────────────────────────────────────────────────────────────
# 2. RAG SYSTEM CLASS
# ─────────────────────────────────────────────────────────────────

class RAGSystem:
    def __init__(
        self,
        chunks_path: str = "../RAG-dataset/concepts.json",
        embedding_model: str = "all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        collection_name: str = "curiochain_math",
        persist_dir: str = "./chroma_db",
    ):
        print("[RAG] Loading chunks...")
        self.chunks = load_chunks(chunks_path)
        print(f"[RAG] Loaded {len(self.chunks)} chunks")

        # Build graph adjacency structure from graph_links
        self._build_graph()

        # Embedding model for baseline retrieval
        print(f"[RAG] Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        # Cross-encoder for re-ranking
        print(f"[RAG] Loading cross-encoder: {cross_encoder_model}")
        self.cross_encoder = CrossEncoder(cross_encoder_model)

        # ChromaDB vector store
        print(f"[RAG] Initializing ChromaDB at {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Index chunks if collection is empty
        if self.collection.count() < len(self.chunks):
            self._index_chunks()

    # ─── GRAPH CONSTRUCTION ─────────────────────────────────────────
    def _build_graph(self):
        """Build in-memory adjacency graph and chunk lookup from graph_links."""
        # Fast lookup: chunk_id → chunk dict
        self.chunk_lookup: Dict[str, Dict] = {
            c["chunk_id"]: c for c in self.chunks
        }

        # Adjacency: chunk_id → {edge_type: [neighbor_chunk_ids]}
        # Skips placeholder targets ([GENERATED: ...]) and non-KB ids.
        self.chunk_graph: Dict[str, Dict[str, List[str]]] = {}

        for chunk in self.chunks:
            cid = chunk["chunk_id"]
            raw_links = chunk.get("graph_links", {})
            if not raw_links:
                continue

            neighbors: Dict[str, List[str]] = {}
            for edge_type, targets in raw_links.items():
                if edge_type in SKIP_EDGE_TYPES:
                    continue
                # Normalise to list
                target_list = targets if isinstance(targets, list) else [targets]
                # Filter out placeholders — keep only real chunk IDs
                real_targets = [
                    t for t in target_list
                    if isinstance(t, str) and not t.startswith("[GENERATED")
                ]
                if real_targets:
                    neighbors[edge_type] = real_targets

            if neighbors:
                self.chunk_graph[cid] = neighbors

        edge_count = sum(
            len(v) for adj in self.chunk_graph.values() for v in adj.values()
        )
        print(f"[RAG] Graph built: {len(self.chunk_graph)} nodes, {edge_count} edges")

    def _index_chunks(self):
        """Embed and index all chunks into ChromaDB."""
        print("[RAG] Indexing chunks into ChromaDB...")
        ids = [c["chunk_id"] for c in self.chunks]
        texts = [c["embed_text"] for c in self.chunks]
        metadatas = [
            {
                "chunk_type": c["chunk_type"],
                "topic": c["topic"],
                "difficulty": c["difficulty"],
            }
            for c in self.chunks
        ]

        # Encode in batches
        batch_size = 64
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            batch_embeddings = self.embedder.encode(batch_texts).tolist()

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metas,
            )

        print(f"[RAG] Indexed {self.collection.count()} chunks")

    # ─── STAGE 1: BASELINE RETRIEVAL (Top-K via vector similarity) ───
    def baseline_retrieval(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve top-k chunks using cosine similarity."""
        query_embedding = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return retrieved

    # ─── STAGE 2: CROSS-ENCODER RE-RANKING (Select Top-N) ───────────
    def rerank(self, query: str, candidates: List[Dict], top_n: int = 3) -> List[Dict]:
        """Re-rank candidates using cross-encoder and return top-n."""
        if not candidates:
            return []

        # Create (query, document) pairs for cross-encoder
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Attach scores and sort
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_n]

    # ─── STAGE 3: GRAPH EXPANSION ────────────────────────────────────
    def graph_expand(
        self,
        seeds: List[Dict],
        max_neighbors: int = 2,
    ) -> List[Dict]:
        """Expand seed chunks by following graph edges to related neighbors.

        Traverses one hop from each seed, prioritising edges that pull in
        foundational context (prerequisites → part_of → demonstrates …).
        Each neighbor inherits a decayed version of its seed's rerank_score
        so the LLM context header can indicate relative confidence.

        Args:
            seeds: Already-retrieved and reranked chunks.
            max_neighbors: Maximum number of graph-expanded chunks to add.

        Returns:
            List of new (non-seed) chunks sourced via graph traversal.
        """
        seen_ids = {c["chunk_id"] for c in seeds}
        expansion: List[Dict] = []

        for seed in seeds:
            if len(expansion) >= max_neighbors:
                break

            cid = seed["chunk_id"]
            adjacency = self.chunk_graph.get(cid, {})
            seed_score = seed.get("rerank_score", 0.0)

            for edge_type in EDGE_PRIORITY:
                if len(expansion) >= max_neighbors:
                    break

                for neighbor_id in adjacency.get(edge_type, []):
                    if neighbor_id in seen_ids:
                        continue
                    if neighbor_id not in self.chunk_lookup:
                        # Chunk referenced in graph but not in KB (cross-topic placeholder)
                        continue

                    seen_ids.add(neighbor_id)
                    nb = self.chunk_lookup[neighbor_id]
                    expansion.append({
                        "chunk_id": neighbor_id,
                        "text": nb["embed_text"],
                        "metadata": {
                            "chunk_type": nb["chunk_type"],
                            "topic": nb["topic"],
                            "difficulty": nb["difficulty"],
                        },
                        # Decayed score — signals lower certainty than direct retrieval
                        "rerank_score": seed_score * GRAPH_SCORE_DECAY,
                        # Provenance metadata (not shown to LLM, useful for debugging)
                        "graph_source": True,
                        "graph_edge": edge_type,
                        "graph_seed": cid,
                    })

                    if len(expansion) >= max_neighbors:
                        break

        return expansion

    # ─── FULL PIPELINE ──────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 3,
        min_score: float = 0.5,
        use_graph: bool = True,
        max_graph_neighbors: int = 2,
    ) -> List[Dict]:
        """Full Graph RAG pipeline: vector retrieval → rerank → score gate → graph expand.

        Args:
            query: The retrieval query string.
            top_k: Number of candidates from vector retrieval (Stage 1).
            top_n: Number of seeds kept after reranking (Stage 2).
            min_score: Minimum rerank score for the top seed. If the best
                       seed scores below this threshold the pipeline returns
                       an empty list — no context is injected. Prevents
                       irrelevant chunks (e.g. score -5.82) from being shown.
            use_graph: Whether to run graph expansion (Stage 3).
            max_graph_neighbors: Max chunks added by graph expansion.

        Returns:
            List of chunks (seeds + graph neighbors) or [] if below threshold.
        """
        # Stage 1: Dense vector retrieval
        candidates = self.baseline_retrieval(query, top_k=top_k)

        # Stage 2: Cross-encoder reranking
        seeds = self.rerank(query, candidates, top_n=top_n)

        # Stage 2.5: Score gate — skip injection for out-of-domain queries
        if not seeds:
            return []

        best_score = seeds[0].get("rerank_score", float("-inf"))
        if best_score < min_score:
            print(
                f"[RAG] Score gate triggered (best={best_score:.3f} < threshold={min_score}). "
                "Skipping context injection."
            )
            return []

        # Keep only individually relevant seeds. This prevents one weakly
        # positive hit from dragging unrelated negative-score chunks into the
        # prompt or into graph expansion.
        filtered_seeds = [
            seed for seed in seeds
            if seed.get("rerank_score", float("-inf")) >= min_score
        ]
        dropped_count = len(seeds) - len(filtered_seeds)
        if dropped_count:
            print(
                f"[RAG] Dropped {dropped_count} low-score seed(s) below threshold={min_score}."
            )

        if not filtered_seeds:
            return []

        # Stage 3: Graph expansion
        if use_graph:
            graph_neighbors = self.graph_expand(filtered_seeds, max_neighbors=max_graph_neighbors)
            if graph_neighbors:
                print(
                    f"[RAG] Graph expansion added {len(graph_neighbors)} neighbor(s): "
                    + ", ".join(
                        f"{c['chunk_id']} (via {c['graph_edge']} from {c['graph_seed']})"
                        for c in graph_neighbors
                    )
                )
            return filtered_seeds + graph_neighbors

        return filtered_seeds

    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a context string for the LLM.

        Chunks sourced via graph expansion are labelled differently so the
        model understands they are structurally related (not directly matched).
        """
        context_parts = []
        for i, c in enumerate(chunks, 1):
            if c.get("graph_source"):
                source_label = (
                    f"graph-expanded via '{c['graph_edge']}' from {c['graph_seed']}"
                )
            else:
                source_label = "direct retrieval"

            context_parts.append(
                f"--- Reference {i} [{c['chunk_id']}] "
                f"(type: {c['metadata']['chunk_type']}, "
                f"topic: {c['metadata']['topic']}, "
                f"source: {source_label}) ---\n"
                f"{c['text']}"
            )
        return "\n\n".join(context_parts)


# ─────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAGSystem()

    test_queries = [
        "How do I find the general term in a binomial expansion?",
        "What is the equation of tangent to a circle?",
        "How to find the sum of an arithmetic geometric progression?",
        # OOD query — should be blocked by score gate
        "How do I find eigenvalues using the characteristic polynomial?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")
        print(f"{'='*60}")

        # Full Graph RAG pipeline
        results = rag.retrieve(q, top_k=10, top_n=3, min_score=0.5, use_graph=True)

        if not results:
            print("[No context injected — query likely out-of-domain]")
            continue

        seeds = [c for c in results if not c.get("graph_source")]
        expanded = [c for c in results if c.get("graph_source")]

        print(f"\n[Seeds ({len(seeds)})]:")
        for c in seeds:
            print(f"  {c['chunk_id']:45s}  rerank={c['rerank_score']:.4f}")

        if expanded:
            print(f"\n[Graph Expanded ({len(expanded)})]:")
            for c in expanded:
                print(
                    f"  {c['chunk_id']:45s}  via '{c['graph_edge']}'"
                    f"  from {c['graph_seed']}"
                )

        context = rag.format_context(results)
        print(f"\n[Context for LLM] ({len(context)} chars)")
