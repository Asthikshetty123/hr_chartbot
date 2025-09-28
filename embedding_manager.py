import os
from typing import List, Dict, Tuple
import pickle

try:
    import faiss
except Exception:
    faiss = None

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class EmbeddingManager:
    """Manages embeddings and a similarity index.

    Behavior:
    - If `sentence_transformers` and `faiss` are available, use them.
    - Otherwise, fall back to a lightweight numpy-based embedding + brute-force search.
    This fallback is intentionally simple and deterministic so debugging works on Windows
    without heavy native dependencies.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.documents = []
        self.index = None
        # default dimension for the transformer model; if fallback used, dimension can vary
        self.dimension = 384

        # Initialize sentence-transformers model if available
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
                # update dimension if model provides it
                if hasattr(self.model, 'get_sentence_embedding_dimension'):
                    try:
                        self.dimension = self.model.get_sentence_embedding_dimension()
                    except Exception:
                        pass
            except Exception:
                self.model = None
        else:
            self.model = None

        # If faiss is available we'll use it for index operations; otherwise, use numpy arrays
        self.use_faiss = faiss is not None

    def create_chunks(self, sections: List[Dict], chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        """Create overlapping chunks from document sections"""
        chunks = []
        chunk_id = 0

        for section in sections:
            text = section.get('content', '')
            words = text.split()

            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)

                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'section_heading': section.get('heading', ''),
                    'section_id': section.get('section_id', ''),
                    'start_word': i,
                    'end_word': min(i + chunk_size, len(words))
                })
                chunk_id += 1

        return chunks

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """A simple deterministic embedding fallback using hashing and trigram counts.

        This is not a replacement for real embeddings, but it allows local testing without
        heavy dependencies.
        """
        embeddings = []
        for t in texts:
            # Lowercase + basic tokenization
            s = t.lower()
            # simple bag-of-characters features: counts of a..z and digits
            vec = np.zeros(384, dtype=np.float32)
            for ch in s:
                idx = ord(ch) % 384
                vec[idx] += 1.0
            # normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.vstack(embeddings)

    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Generate embeddings for all chunks; use real model if available, otherwise fallback."""
        texts = [chunk['text'] for chunk in chunks]
        if self.model is not None:
            try:
                emb = self.model.encode(texts, normalize_embeddings=True)
                emb = np.asarray(emb, dtype=np.float32)
                return emb
            except Exception:
                # fall through to fallback
                pass

        return self._fallback_embed(texts)

    def build_faiss_index(self, embeddings: np.ndarray, chunks: List[Dict]):
        """Build index (FAISS when available, else store numpy arrays)."""
        self.dimension = embeddings.shape[1]
        self.documents = chunks

        if self.use_faiss:
            try:
                # FAISS expects float32
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(embeddings.astype('float32'))
                return
            except Exception:
                # if faiss fails at runtime, fallback to numpy
                self.index = None
                self.use_faiss = False

        # fallback: store embeddings as numpy matrix
        self._np_embeddings = np.asarray(embeddings, dtype=np.float32)

    def save_index(self, filepath: str):
        """Save FAISS index and documents (or numpy fallback)."""
        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, f"{filepath}.index")
        else:
            # save numpy embeddings
            np.save(f"{filepath}.npy", getattr(self, '_np_embeddings', np.zeros((0, self.dimension), dtype=np.float32)))

        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self.documents, f)

    def load_index(self, filepath: str):
        """Load FAISS index and documents (or numpy fallback)."""
        if self.use_faiss:
            try:
                self.index = faiss.read_index(f"{filepath}.index")
            except Exception:
                self.index = None
                self.use_faiss = False

        if not self.use_faiss:
            try:
                self._np_embeddings = np.load(f"{filepath}.npy")
                self.dimension = self._np_embeddings.shape[1]
            except Exception:
                self._np_embeddings = np.zeros((0, self.dimension), dtype=np.float32)

        try:
            with open(f"{filepath}.pkl", 'rb') as f:
                self.documents = pickle.load(f)
        except Exception:
            self.documents = []

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Perform similarity search using FAISS if available, otherwise numpy brute-force."""
        # Create query embedding
        if self.model is not None:
            try:
                q_emb = self.model.encode([query], normalize_embeddings=True)
                q = np.asarray(q_emb, dtype=np.float32)
            except Exception:
                q = self._fallback_embed([query])
        else:
            q = self._fallback_embed([query])

        # FAISS search
        if self.use_faiss and self.index is not None:
            try:
                scores, indices = self.index.search(q.astype('float32'), k)
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.documents):
                        results.append((self.documents[idx], float(scores[0][i])))
                return results
            except Exception:
                # fall back to numpy
                pass

        # Numpy brute-force cosine-similarity
        emb_matrix = getattr(self, '_np_embeddings', None)
        if emb_matrix is None or emb_matrix.size == 0:
            return []

        qv = q[0].astype(np.float32)
        # cosine similarity since embeddings were normalized in fallback
        dots = emb_matrix @ qv
        # get top-k
        idxs = np.argsort(-dots)[:k]
        results = []
        for idx in idxs:
            if idx < len(self.documents):
                results.append((self.documents[idx], float(dots[idx])))
        return results