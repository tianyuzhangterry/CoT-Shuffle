# CoT-Shuffle
CoT-Shuffle: Semantic Reconstruction for Chain-of-Thought Compression in Large Language Models

CoT-Shuffle is a three‑layer Chain‑of‑Thought (CoT) compression framework that use semantic reconstruction instead of token trimming.

```
Full CoT  ──►  Segmentation  ──►  Scoring / Layer Routing  ──►  Prompt‑based Trim  ──►  Compressed CoT
                                    │                           │
                                    └──►  L1   (Light)          │  – keep synonyms + numbers
                                         L2   (Medium)         – template 2‑3 steps
                                         L3   (Heavy)          – single sentence / [SKIP]
```
### Typical Output

```
{Question:} Kylar went to the store to buy glasses for his new apartment. One glass costs \$5, but every second glass costs only 60\% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?\\

{Original CoT:} The discount price of one glass is 60/100 * 5 = \$<<60/100*5=3>>3. If every second glass is cheaper, that means Kylar is going to buy 16 / 2 = <<16/2=8>>8 cheaper glasses. So for the cheaper glasses, Kylar is going to pay 8 * 3 = \$<<8*3=24>>24. And for the regular-priced glasses, Kylar will pay 8 * 5 = \$<<8*5=40>>40. So in total Kylar needs to pay 24 + 40 = \$<<24+40=64>>64 for the glasses he wants to buy. \\

{Ours:} Step 1: 8 * 3 = 24. Step 2: 8 * 5 = 40. Therefore, Kylar needs to pay \$24 + \$40 = \$64 for the glasses he wants to buy. \\

```

---

## Repository Layout

```
├─ prompts/                 # light_prompt.md / medium_prompt.md / heavy_prompt.md
├─ cot_compressor.py        # main pipeline (segment → route → trim)
├─ solve_with_cot.py        # question + compressed chain → answer
├─ complexity_estimator.py  # DistilBERT‑based difficulty score
├─ cot_semantic_segmenter.py# DistilBERT sentence embedding splitter
└─ README.md                # you are here
```

---

## How It Works

1. **Semantic Segmentation**
      `cot_semantic_segmenter.py` splits long CoT when cosine < τ.
2. **Difficulty Scoring**
      `complexity_estimator.py` assigns an overall and per‑chunk score.
3. **Layer Routing**
      Numeric lines + tail lines receive bonus, then z‑score → L1/L2/L3.
4. **Prompt‑based Processing**
      Qwen Instruct + three system prompts generate compressed text.
5. **Evaluating**
      Testing with authorized evaluation metrics

---


## License

Apache 2.0
