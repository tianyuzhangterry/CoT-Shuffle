from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Prompt template — single Python string with explicit \n line breaks
_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful assistant. Use the reasoning chain to answer.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Question: {question}\n\n"
    "Reasoning chain:\n{cot}\n\n"
    "Give the final answer in a clear sentence.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

class CoTAnswerer:
    """Hold a Qwen pipeline and answer questions using compressed CoT."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: int | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        dev = 0 if torch.cuda.is_available() else -1
        self.device = dev if device is None else device
        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer, device=self.device)

    # Build prompt with question + compressed chain
    def _build_prompt(self, question: str, cot: str) -> str:
        return _TEMPLATE.format(question=question.strip(), cot=cot.strip())

    def answer(self, question: str, cot: str, max_tokens: int = 64) -> str:
        """Return model answer given *question* and *cot*."""
        prompt = self._build_prompt(question, cot)
        out = self.pipeline(prompt, max_new_tokens=max_tokens, temperature=0.0, do_sample=False, return_full_text=False)[0]
        return out["generated_text"].strip()

# ------------------ demo ------------------
if __name__ == "__main__":
    q = (
        '''Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
        '''
    )

    short_cot = (
        '''Step 1: Convert 50 minutes to hours by dividing by 60. 50 ÷ 60 = 0.8333 (50 minutes) To convert minutes to hours, divide the number of minutes by 60. So, 50 minutes divided by 60 equals... To find her earnings, multiply the hours worked (0.8333) by her hourly rate ($12). 50 minutes divided by 60 is 5/6, or roughly 0.8333 hours. 0.8333 times 12 equals her earnings. 0.8 times 12 is 9.6, and 0.0333 times 12 is about 0.4, so adding them gives roughly $10. Step 2: 50 minutes is exactly \( \frac{5}{6} \) of an hour. \( \frac{5}{6} \) times 12 equals \( 5 \times 12 = 60 \), then divided by 6 is 10. Weng earned $10. Since we're dealing with dollars and cents, and the calculation came out to an even $10, I don't need to round. Therefore, Weng earned $10 yesterday for babysitting.'''
    )

    solver = CoTAnswerer()
    print("[Answer]", solver.answer(q, short_cot))
