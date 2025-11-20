import os, re, json, time, argparse, random
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from fractions import Fraction
from decimal import Decimal, InvalidOperation

# =================== 配置 ===================
SEED = 42
ABS_TOL, REL_TOL = 1e-9, 1e-6      # 数值判定容差
DEF_MAX_NEW = 64
DEF_INPUT = "processed_data/mixchain_gsm8k_train_compressed.csv"
DEF_OUT_CSV = "answers/llm_extracted.csv"
DEF_OUT_METRICS = "answers/metrics_llm.json"

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =================== 规范化与解析工具 ===================
def _normalize_tex_simple(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().replace("−", "-").replace("\\,", "").replace("\\!", "")
    s = s.replace("\\left","").replace("\\right","").replace("$","")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    m = re.fullmatch(r"\\boxed\{(.+)\}", s)
    if m: s = s[m.start(1):m.end(1)]
    return s.strip()

def _parse_scalar_num(s: str):
    """只解析纯标量（浮点/科学、a/b、\\frac{a}{b}、百分号）。否则返回 None。"""
    t = _normalize_tex_simple(s)
    if not t: return None
    is_pct = t.endswith("%")
    if is_pct: t = t[:-1].strip()

    m = re.fullmatch(r"\\frac\{\s*([-+]?\d+)\s*\}\{\s*([-+]?\d+)\s*\}", t)
    if m:
        den = int(m.group(2))
        if den == 0: return None
        val = int(m.group(1)) / den
        return val/100.0 if is_pct else val

    if re.fullmatch(r"[-+]?\d+\s*/\s*\d+", t):
        try:
            val = float(Fraction(t))
            return val/100.0 if is_pct else val
        except Exception:
            return None

    try:
        val = float(Decimal(t.replace(",", "")))
        return val/100.0 if is_pct else val
    except InvalidOperation:
        return None

def eq_num(a, b, abs_tol=ABS_TOL, rel_tol=REL_TOL):
    if a is None or b is None: return False
    return abs(a - b) <= max(abs_tol, rel_tol * max(1.0, abs(b)))

def math_equal_loose(pred: str, gold: str) -> bool:
    """统一判分：先数值；再字符串轻规整；最后 SymPy 等价（含复数 i）。"""
    P, A = str(pred), str(gold)
    vp, va = _parse_scalar_num(P), _parse_scalar_num(A)
    if vp is not None and va is not None:
        return eq_num(vp, va)
    if _normalize_tex_simple(P) == _normalize_tex_simple(A):
        return True
    # SymPy 等价（有限兜底）
    try:
        import sympy as sp
        def prep(s: str) -> str:
            s = _normalize_tex_simple(s)
            s = s.replace(r"\cdot","*").replace(r"\times","*")
            s = s.replace("^","**")
            s = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", s)
            s = re.sub(r"\\frac\{\s*([^}]*)\s*\}\{\s*([^}]*)\s*\}", r"(\1)/(\2)", s)
            s = s.replace(r"\pi", "pi")
            s = re.sub(r"(?<=\d)\s*i\b", "*I", s)            # 9i -> 9*I
            s = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "I", s) # 独立 i -> I
            return s
        ps, gs = prep(P), prep(A)
        # 等式情况：lhs=rhs
        def to_expr(expr_str: str):
            if "=" in expr_str:
                parts = expr_str.split("=")
                if len(parts)==2:
                    return sp.sympify(parts[0]), sp.sympify(parts[1]), True
            return sp.sympify(expr_str), sp.Integer(0), False
        lp, rp, peq = to_expr(ps)
        lg, rg, geq = to_expr(gs)
        if peq and geq:
            return sp.simplify((lp - rp) - (lg - rg)) == 0
        elif not peq and not geq:
            diff = sp.simplify(lp - lg)
            if diff.is_Number:
                try: return eq_num(float(diff.evalf()), 0.0)
                except: return diff == 0
            return diff == 0
        else:
            if peq and not geq:
                return sp.simplify((lp - rp) - lg) == 0
            if geq and not peq:
                return sp.simplify(lp - (lg - rg)) == 0
            return False
    except Exception:
        return False

# =================== LLM 抽取器 ===================
_EXTRACT_SYSTEM = (
    "你正在解数学题，你只用从提供的 compressed reasoning chain 中提取题目的最终答案。\n"
    "严格遵守：\n"
    "1) 只阅读给定的 chain，本题答案必须从原文中【直接拷贝原样片段】（verbatim）。\n"
    "2) 保持原有 LaTeX 形式（例如 \\sqrt{}, \\frac{}{}, \\pi, \\boxed{}），不要改写为 sqrt()、*、小数、添加变量名或等号。\n"
    "3) 只返回最终答案，并且必须只用 <ans>...</ans> 包裹；不要输出任何额外文字。\n"
)

_EXTRACT_USER_TMPL = (
    "Compressed chain（仅供抽取，不要重写）：\n```\n{cot}\n```\n"
    "请从上面的文本中【原样复制】最终答案（优先取最后一句中的 \\boxed{...}；否则取最后一句中倒数第一对 $...$；"
    "否则取最后一句中 'equal to/equals' 之后的表达式；否则取最后一句中最后一个 '=' 右侧的表达式），"
    "并用 <ans>...</ans> 返回。严禁改写格式或添加内容。"
)

ANS_TAG_RE = re.compile(r"<ans>(.*?)</ans>", flags=re.DOTALL)

class AnswerExtractor:
    def __init__(self, model_path: str, tokenizer_path: str | None = None,
                 adapter_path: str | None = None, max_new_tokens: int = DEF_MAX_NEW):
        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        tok_name = tokenizer_path if tokenizer_path else model_path

        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            try: self.model = self.model.merge_and_unload()
            except Exception: pass
        self.model.eval()
        self.pipe = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)
        self.max_new_tokens = max_new_tokens

    def _build_prompt(self, cot: str) -> str:
        cot_text = str(cot).strip()  # 不做任何 { } 转义，直接原样放入
        user_content = (
            "Compressed chain（仅供抽取，不要重写）：\n"
            "```\n" + cot_text + "\n```\n"
            "请从上面的文本中【原样复制】最终答案（优先取最后一句中的 \\boxed{...}；"
            "否则取最后一句中倒数第一对 $...$；"
            "否则取最后一句中 'equal to/equals' 之后的表达式；"
            "否则取最后一句中最后一个 '=' 右侧的表达式），"
            "并用 <ans>...</ans> 返回。严禁改写格式或添加内容。"
        )
        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def extract(self, cot: str) -> str:
        prompt = self._build_prompt(cot)
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0, do_sample=False, return_full_text=False
        )[0]["generated_text"]

        def _trim_ans(s: str) -> str:
            s = s.strip()
            if s.startswith("$") and s.endswith("$") and len(s) >= 2:
                s = s[1:-1].strip()
            return s

        m = ANS_TAG_RE.search(out)
        return _trim_ans(m.group(1) if m else out)

# =================== 主流程 ===================
def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--input", type=str, default=DEF_INPUT)
    ap.add_argument("--out_csv", type=str, default=DEF_OUT_CSV)
    ap.add_argument("--out_metrics", type=str, default=DEF_OUT_METRICS)
    ap.add_argument("--cot_column", type=str, default="compressed_chain")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=DEF_MAX_NEW)
    ap.add_argument("--model_path", type=str, default="/mlx/users/tianyuzhang/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--tokenizer_path", type=str, default=None)
    ap.add_argument("--adapter_path", type=str, default=None)
    args = ap.parse_args()

    set_seed()

    # 读数据
    df = pd.read_csv(args.input)
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    if args.cot_column not in df.columns:
        raise ValueError(f"缺少列 `{args.cot_column}`")

    # gold：文本 + 数值
    df["gold_text"] = None
    df["gold_num"] = np.nan
    if "target" in df.columns:
        df["gold_text"] = df["target"].astype(str)
    elif "answer" in df.columns:
        df["gold_text"] = df["answer"].astype(str)
    else:
        print("[warn] 未找到 target/answer 列；gold_text 将为空。")

    # 数值化 gold
    def _gold_num_from_row(x):
        try: return _parse_scalar_num(x)
        except: return None
    if df["gold_text"].notna().any():
        gn = df["gold_text"].apply(_gold_num_from_row)
        df["gold_num"] = pd.to_numeric(gn, errors="coerce")

    # 初始化抽取器
    extractor = AnswerExtractor(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens
    )
    # 另一个 tokenizer 仅用于 token 统计
    tok_for_count = extractor.tokenizer

    # 评测
    answers_text, answers_num = [], []
    corrects, latencies = [], []
    gen_all_tok_counts = []

    use_cuda = torch.cuda.is_available()
    t0_all = time.time()

    for i, row in df.iterrows():
        cot = str(row[args.cot_column])

        try:
            if use_cuda: torch.cuda.synchronize()
            t0 = time.time()
            ans_text = extractor.extract(cot)   # ★ 仅基于 compressed_chain 抽取
            if use_cuda: torch.cuda.synchronize()
            dt = time.time() - t0
        except Exception as e:
            ans_text = ""
            dt = float("nan")
            print(f"[warn] 抽取失败 @ {i}: {e}")

        # 生成 tokens（抽取结果的长度，作为生成侧输出 token 的近似；若你想统计解码真实 tokens，需要 pipeline 输出 token-ids）
        gen_all_tok_counts.append(len(tok_for_count(ans_text, add_special_tokens=False).input_ids))

        # 数值化（仅纯标量）
        pnum = _parse_scalar_num(ans_text)

        gold_text = (row.get("gold_text", "") or "")
        gold_num  = row.get("gold_num", None)

        # 1) 数值等价（-5/9, \frac{14}{3}, 百分号等）
        ok_num = eq_num(pnum, gold_num)

        # 2) 符号/等式等价（含复数 i、等式 lhs=rhs）
        ok_math = math_equal_loose(ans_text, gold_text) if gold_text else False

        # 3) 子串包含：只要正确内容（gold_text）出现在 model_answer 里就算对
        def _compact_tex(s: str) -> str:
            s = s or ""
            s = s.strip().replace("−", "-").replace("\\,", "").replace("\\!", "")
            s = s.replace("\\left","").replace("\\right","").replace("$","")
            s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
            return s.replace(" ", "")

        ok_substring = False
        if gold_text:
            ok_substring = _compact_tex(gold_text) in _compact_tex(ans_text)

        ok = bool(ok_num or ok_math or ok_substring)
        answers_text.append(ans_text)
        answers_num.append(pnum)
        corrects.append(ok)
        latencies.append(dt)

        if (i+1) % 20 == 0:        
            acc = float(np.mean(corrects))
            avg_lat = float(np.nanmean(latencies))
            print(f"[{i+1}/{len(df)}] acc={acc:.3f}, avg_lat={avg_lat:.2f}s")

    total_time = time.time() - t0_all

    # 写 CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df["model_answer"] = answers_text
    df["model_answer_num"] = answers_num
    df["correct"] = corrects
    df["latency_s"] = latencies
    df["gen_all_tokens"] = gen_all_tok_counts
    df.to_csv(args.out_csv, index=False, encoding="utf-8")

    # metrics
    metrics = {
        "n_samples": int(len(df)),
        "accuracy": float(np.mean(corrects)) if len(corrects) else 0.0,
        "avg_latency_sec": float(np.nanmean(latencies)) if len(latencies) else 0.0,
        "total_time_sec": float(total_time),
        "avg_gen_tokens": float(np.mean(gen_all_tok_counts)) if len(gen_all_tok_counts) else 0.0,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "tokenizer_path": args.tokenizer_path or args.model_path,
        "cot_column": args.cot_column,
        "max_new_tokens": args.max_new_tokens,
        "input_path": args.input,
        "out_csv": args.out_csv,
    }
    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n=== Evaluation (LLM extraction) Summary ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Details saved to: {args.out_csv}")
    print(f"Metrics saved to: {args.out_metrics}")

if __name__ == "__main__":
    main()
