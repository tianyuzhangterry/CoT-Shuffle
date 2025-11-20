"""semantic_segmenter_distilbert.py

Segment a Chain‑of‑Thought string into semantic blocks **using DistilBERT
embeddings**.

Idea
----
1.  First split text into coarse sentences with regex / spaCy‑mini (optional).
2.  Encode each sentence with `distilbert-base-uncased` [CLS] embedding.
3.  Compute cosine similarity between consecutive sentences.
4.  Break a segment whenever similarity < THRESH (default 0.75).

This adapts automatically: tightly related sentences stay together; topic
shifts/logic leaps create new segments.
"""

from __future__ import annotations
from typing import List
import re
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class DistilBERTSegmenter:
    def __init__(
        self, model_name: str = "distilbert-base-uncased", device: str | None = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed(self, sentences: List[str]) -> torch.Tensor:
        """Return (n, 768) L2‑normalised embeddings."""
        ids = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        out = self.model(**ids).last_hidden_state[:, 0]  # [CLS] vector
        return F.normalize(out, dim=1)

    def segment(self, text: str, thresh: float = 0.9, max_len: int = 50) -> List[str]:
        """Return semantic blocks based on similarity threshold."""
        # 1. coarse sentence split
        sents = [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]
        if not sents:
            return [text.strip()]
        # 2. embeddings
        vecs = self.embed(sents)
        # 3. cosine sim between consecutive sentences
        sims = (vecs[:-1] * vecs[1:]).sum(dim=1).cpu()
        segments = [sents[0]]
        for i, sim in enumerate(sims, 1):   
            if sim < thresh or len(segments[-1].split()) > max_len:
                segments.append(sents[i])
            else:
                segments[-1] += " " + sents[i]
        return segments


# -------- demo --------
if __name__ == "__main__":
    demo = """
            Let's tackle this problem step by step. Natalia sold clips to 48 of her friends in April. 
            Then, in May, she sold half as many clips as she did in April. 

            To find out how many clips she sold altogether in these two months, I first need to calculate her May sales. 
            Since May’s sales were half of April’s, I divide 48 by 2, which gives 24. 
            That means she sold 24 clips in May.

            Adding the April and May sales together, 48 plus 24 equals 72.

            To double-check, if she sold 48 in April and half of that (24) in May, the total is indeed 72. 
            Thinking of it another way, total sales are April's amount plus half of April's amount—
            48 plus 0.5 times 48 equals 72.

            Alternatively, if May's sales are half of April's, then the total is 1.5 times April's sales. 
            So, 48 times 1.5 equals 72.

            All methods give the same result. I’m confident with this answer: 
            Natalia sold a total of 72 clips in April and May altogether.

        """
    seg = DistilBERTSegmenter()
    for i, block in enumerate(seg.segment(demo), 1):
        print(f"S{i}: {block}")
