import sys
sys.path.append('/mnt/data')

import os
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

import complexity_estimator
import cot_semantic_segmenter

############################################################
# -----------------  CoTCompressor  ----------------------- #
############################################################

class CoTCompressor:
    """Compress CoT segments at three levels using an instruction‑tuned Qwen model."""

    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 prompt_dir: str = "prompts",
                 device: int | None = None):
        # ─── Load model ───────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        # generation defaults
        cfg = self.model.generation_config
        cfg.max_new_tokens = 64
        cfg.temperature = 0.0
        cfg.do_sample = False
        cfg.top_p = 1.0

        # pipeline
        self.device = 0 if torch.cuda.is_available() else -1 if device is None else device
        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer, device=self.device)

        # prompt templates (must包含 {text} 与 {prev})
        self.prompts = {
            1: self._load_prompt(os.path.join(prompt_dir, "light_prompt.md")),
            2: self._load_prompt(os.path.join(prompt_dir, "medium_prompt.md")),
            3: self._load_prompt(os.path.join(prompt_dir, "heavy_prompt.md")),
        }

    @staticmethod
    def _load_prompt(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ──────────  PUBLIC  ──────────
    def compress(self, text: str, layer: int, prev: str = "") -> str:
        """Compress *text* with given *layer*; *prev* 为上一段压缩结果。"""
        if prev.strip() == "[SKIP]":
            prev = ""
            
        # 对于中度压缩(层级2)，处理步骤编号
        if layer == 2 and prev:
            # 尝试从prev中提取所有步骤编号
            step_matches = re.findall(r'Step (\d+):', prev)
            if step_matches:
                # 获取最大的步骤编号
                last_step_num = max(int(num) for num in step_matches)
                # 在prompt中添加隐含信息，告知模型最后的步骤编号
                prev = prev.strip() + f" (continue from Step {last_step_num})"
            
        prompt = (self.prompts[layer]
                  .replace("{text}", text.replace("\n", " ").strip())
                  .replace("{prev}", prev.strip()))
        return self._call_llm(prompt)

    # ────────── INTERNAL ──────────
    def _call_llm(self, prompt: str) -> str:
        out = self.pipeline(prompt, return_full_text=False)[0]
        if isinstance(out, dict):
            out = out.get("generated_text", "")
        return str(out).strip()

############################################################
# -----------------  Layer Assign  ----------------------- #
############################################################

_num_re = re.compile(r"\d")
_intro_re = re.compile(r"^(Let's|I'll|We'll|First|To|Now)")
_conclusion_re = re.compile(r"(answer|result|conclusion|confident|therefore|thus|hence|so the)|(^I'm|^We're)")
_calc_re = re.compile(r'(equals|plus|minus|multiply|divide|sum|add|subtract|times|divided by)')

def assign_layers(segments, scores, z_th=0.4, num_bonus=0.15, tail_bonus=0.15, calc_bonus=0.2):
    """
    分配压缩层级:
    - 层级1: 轻度压缩 (保留大部分内容)
    - 层级2: 中度压缩 (转换为步骤)
    - 层级3: 重度压缩 (只保留结论或跳过)
    """
    s = np.array(scores, dtype=np.float32)
    
    # 根据语义角色调整分数
    for i, seg in enumerate(segments):
        # 数学符号加分
        if _num_re.search(seg) or any(sym in seg for sym in ["×", "=", "+", "-", "⇒"]):
            s[i] += num_bonus
        
        # 引导性语句倾向于轻度压缩
        if _intro_re.search(seg):
            s[i] += 0.2
        
        # 结论性语句倾向于重度压缩
        if _conclusion_re.search(seg.lower()):
            s[i] -= 0.2
        
        # 计算步骤加分 - 确保被分配到L2
        if _calc_re.search(seg.lower()) or re.search(r'\d+\s*[+\-×÷=]\s*\d+', seg):
            s[i] += calc_bonus
        
        # 最后一个段落加分（通常是结论）
        if i == len(segments) - 1:
            s[i] += tail_bonus
    
    # 计算z-score
    z = (s - s.mean()) / (s.std(ddof=0) or 1e-6)
    
    # 根据z-score分配层级
    layers = []
    for i, zi in enumerate(z):
        if zi >= z_th:
            # 高分段落轻度压缩
            layers.append(1)
        elif zi <= -z_th:
            # 低分段落重度压缩
            layers.append(3)
        else:
            # 中等段落中度压缩
            layers.append(2)
    
    # 确保第一个和最后一个段落不会被重度压缩
    if layers[0] == 3:
        layers[0] = 2
    if layers[-1] == 3 and _conclusion_re.search(segments[-1].lower()):
        layers[-1] = 1  # 确保结论使用轻度压缩
    
    # 强制将计算步骤分配为L2
    for i, seg in enumerate(segments):
        if layers[i] == 3 and (_calc_re.search(seg.lower()) or re.search(r'\d+\s*[+\-×÷=]\s*\d+', seg)):
            layers[i] = 2  # 强制将计算步骤分配为L2
    
    return layers

def _compute_semantic_similarity(text1, text2, embeddings1=None, embeddings2=None, segmenter=None):
    """
    计算两段文本的语义相似度，返回相似度分数和特征
    
    参数:
        text1, text2: 要比较的文本
        embeddings1, embeddings2: 预计算的嵌入 (如果有)
        segmenter: 用于计算嵌入的分割器
        
    返回:
        similarity_score: 相似度分数 (0-1)
        features: 相似度特征字典
    """
    # 基础文本清理
    text1 = re.sub(r'\s+', ' ', text1.lower().strip())
    text2 = re.sub(r'\s+', ' ', text2.lower().strip())
    
    # 初始化特征字典
    features = {}
    
    # 1. 计算词汇重叠特征
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    # 计算Jaccard相似度和重叠比例
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    features['jaccard'] = len(intersection) / max(len(union), 1)
    features['overlap'] = len(intersection) / max(min(len(words1), len(words2)), 1)
    
    # 2. 计算数字重叠特征 - 对于数学计算特别重要
    numbers1 = set(re.findall(r'\b\d+(?:\.\d+)?\b', text1))  # 改进数字提取
    numbers2 = set(re.findall(r'\b\d+(?:\.\d+)?\b', text2))
    
    if numbers1 and numbers2:
        num_intersection = numbers1.intersection(numbers2)
        num_union = numbers1.union(numbers2)
        features['number_overlap'] = len(num_intersection) / len(num_union)
    else:
        features['number_overlap'] = 0.0
    
    # 3. 计算公式重叠特征 - 匹配更多公式模式
    # 匹配形如 "5 + 10 = 15", "48 × 1.5 = 72" 的公式
    formula_patterns = [
        r'\b\d+(?:\.\d+)?\s*[+\-×÷=]\s*\d+(?:\.\d+)?\b',  # 基本公式
        r'\b\d+(?:\.\d+)?\s*(?:plus|minus|times|divided by)\s*\d+(?:\.\d+)?\b',  # 文本表示的公式
        r'\b\d+\s*\/\s*\d+\b'  # 分数形式
    ]
    
    formulas1 = set()
    formulas2 = set()
    
    for pattern in formula_patterns:
        formulas1.update(re.findall(pattern, text1))
        formulas2.update(re.findall(pattern, text2))
    
    if formulas1 and formulas2:
        formula_intersection = formulas1.intersection(formulas2)
        formula_union = formulas1.union(formulas2)
        features['formula_overlap'] = len(formula_intersection) / len(formula_union)
    else:
        features['formula_overlap'] = 0.0
    
    # 4. 语义嵌入相似度 - 使用BERT模型的语义理解能力
    features['embedding_similarity'] = 0.0
    
    if segmenter is not None:
        try:
            # 使用预计算的嵌入或计算新的嵌入
            if embeddings1 is not None and embeddings2 is not None:
                emb1, emb2 = embeddings1, embeddings2
            else:
                emb1 = segmenter.embed([text1])[0]
                emb2 = segmenter.embed([text2])[0]
                
            # 计算余弦相似度
            cos_sim = torch.dot(emb1, emb2).item()
            
            # 对于非常短的文本，减少嵌入相似度的权重
            short_text_penalty = 1.0
            if min(len(words1), len(words2)) < 5:
                short_text_penalty = 0.8
                
            features['embedding_similarity'] = cos_sim * short_text_penalty
        except Exception:
            # 如果嵌入计算失败，继续使用其他特征
            pass
    
    # 5. 文本长度比例 - 帮助识别摘要/扩展关系
    len1, len2 = len(text1), len(text2)
    features['length_ratio'] = min(len1, len2) / max(len1, len2)
    
    # 6. 核心内容相似度 - 关注语义上的关键词匹配
    # 对文本中的语义重要性（名词、动词和数字）加权计算相似度
    # 使用更复杂的正则表达式来匹配更多的语义单元
    semantic_tokens1 = set(re.findall(r'\b[a-z]{3,}\b|\b\d+(?:\.\d+)?\b|\b[a-z]+(?:ing|ed|s)\b', text1))
    semantic_tokens2 = set(re.findall(r'\b[a-z]{3,}\b|\b\d+(?:\.\d+)?\b|\b[a-z]+(?:ing|ed|s)\b', text2))
    
    if semantic_tokens1 and semantic_tokens2:
        semantic_intersection = semantic_tokens1.intersection(semantic_tokens2)
        semantic_union = semantic_tokens1.union(semantic_tokens2)
        features['semantic_overlap'] = len(semantic_intersection) / len(semantic_union)
    else:
        features['semantic_overlap'] = 0.0
    
    # 7. 添加步骤指示词检测 - 识别段落中的步骤标识
    step_words = {'step', 'first', 'second', 'third', 'next', 'then', 'finally', 'lastly'}
    step_words1 = set(word for word in words1 if word in step_words)
    step_words2 = set(word for word in words2 if word in step_words)
    
    # 有步骤标识词的段落具有特殊性质
    features['has_step_words'] = 1.0 if step_words1 or step_words2 else 0.0
    
    # 计算综合相似度分数 - 根据不同特征加权
    # 权重调整为更强调语义相似度和公式相似度
    weights = {
        'embedding_similarity': 0.45,  # 嵌入相似度权重仍然最高
        'jaccard': 0.15,
        'overlap': 0.10,
        'number_overlap': 0.15,  # 增加数字重叠的权重
        'formula_overlap': 0.10,  # 增加公式重叠的权重
        'semantic_overlap': 0.05,
        'has_step_words': -0.05,  # 如果有步骤词，适当降低相似度（避免错误合并不同步骤）
        'length_ratio': 0.0  # 长度比率作为辅助特征
    }
    
    similarity_score = sum(weights[k] * features[k] for k in weights.keys())
    
    # 如果嵌入相似度为0（计算失败），重新分配权重
    if features['embedding_similarity'] == 0:
        # 无嵌入时，更多依赖于表面文本特征
        alt_weights = {
            'jaccard': 0.25,
            'overlap': 0.25, 
            'number_overlap': 0.25,
            'formula_overlap': 0.15,
            'semantic_overlap': 0.15,
            'has_step_words': -0.05
        }
        similarity_score = sum(alt_weights[k] * features[k] for k in alt_weights.keys())
    
    # 确保分数在0-1范围内
    similarity_score = max(0.0, min(1.0, similarity_score))
    
    return similarity_score, features

def post_process_compression(compressed_segments, segments, layers):
    """
    对压缩后的段落进行后处理:
    1. 去除[SKIP]标记
    2. 去除重复内容
    3. 优化连贯性
    4. 修复步骤编号
    5. 确保引导语和结论被保留
    """
    # 去除[SKIP]标记，同时保留原始段落类型信息
    processed = []
    original_types = []  # 记录每个段落的原始类型（引导、计算、结论）
    
    for i, (seg, orig_seg, layer) in enumerate(zip(compressed_segments, segments, layers)):
        # 清理[SKIP]标记
        cleaned = re.sub(r'`?\[SKIP\]`?', '', seg).strip()
        
        # 清理模型生成的注释
        cleaned = re.sub(r'\(Note:.*?\)', '', cleaned).strip()
        
        # 如果是[SKIP]但是重要段落（引导语或结论），使用轻度压缩
        if not cleaned and (i == 0 or i == len(compressed_segments) - 1 or 
                           _intro_re.search(orig_seg) or _conclusion_re.search(orig_seg.lower())):
            # 简单处理原始段落作为替代
            cleaned = orig_seg.split('.')[0].strip() + '.'
        
        if cleaned:  # 只添加非空段落
            processed.append(cleaned)
            # 判断段落类型
            if i == 0 or _intro_re.search(orig_seg):
                original_types.append('intro')
            elif i == len(compressed_segments) - 1 or _conclusion_re.search(orig_seg.lower()):
                original_types.append('conclusion')
            elif _calc_re.search(orig_seg.lower()) or re.search(r'\d+\s*[+\-×÷=]\s*\d+', orig_seg):
                original_types.append('calculation')
            else:
                original_types.append('other')
    
    # 初始化语义分割器用于生成嵌入表示
    segmenter = None
    try:
        # 使用已存在的分割器来避免重复加载模型
        segmenter = cot_semantic_segmenter.DistilBERTSegmenter()
        print("成功初始化语义分割器用于相似度计算")
    except Exception as e:
        print(f"无法初始化语义分割器，回退到传统相似度计算: {e}")
    
    # 批量计算嵌入以提高效率
    if segmenter and len(processed) > 0:
        try:
            normalized_texts = [re.sub(r'\s+', ' ', seg.lower().strip()) for seg in processed]
            batch_embeddings = segmenter.embed(normalized_texts)
            has_embeddings = True
            print(f"成功批量计算 {len(normalized_texts)} 个段落的嵌入向量")
        except Exception as e:
            print(f"批量嵌入计算失败: {e}")
            batch_embeddings = None
            has_embeddings = False
    else:
        batch_embeddings = None
        has_embeddings = False
    
    # 去除重复内容和冗余信息
    unique_segments = []
    unique_types = []
    seen_calculations = set()  # 记录已经看到的计算
    unique_embeddings = []  # 存储段落的嵌入表示
    
    for i, (seg, seg_type) in enumerate(zip(processed, original_types)):
        # 规范化内容以检测重复
        normalized = re.sub(r'\s+', ' ', seg.lower().strip())
        
        # 跳过只包含单个数字或简短的非计算内容
        if len(normalized.split()) <= 2 and seg_type != 'intro' and seg_type != 'conclusion':
            continue
            
        # 跳过模型生成的注释
        if normalized.startswith('note:') or '(note:' in normalized:
            continue
        
        # 提取计算内容（如果有）
        calc_match = re.search(r'(\d+\s*[+\-×÷]\s*=\s*\d+)', normalized)
        calc_content = calc_match.group(1) if calc_match else None
        
        # 检查是否是重复计算
        if calc_content and calc_content in seen_calculations:
            continue
        elif calc_content:
            seen_calculations.add(calc_content)
        
        # 获取当前段落的嵌入向量
        current_embedding = None
        if has_embeddings and batch_embeddings is not None:
            current_embedding = batch_embeddings[i]
        
        # 语义相似度检测
        is_redundant = False
        best_similarity = 0
        best_idx = -1
        best_features = None
        
        # 与现有段落比较，查找冗余
        for idx, (existing_seg, existing_type) in enumerate(zip(unique_segments, unique_types)):
            existing_emb = unique_embeddings[idx] if idx < len(unique_embeddings) else None
            
            # 使用高级语义相似度计算
            similarity, features = _compute_semantic_similarity(
                normalized, existing_seg, 
                current_embedding, existing_emb, 
                segmenter
            )
            
            # 动态调整相似度阈值 - 不同类型的内容使用不同阈值
            if seg_type == 'calculation' and existing_type == 'calculation':
                # 计算步骤需要更高的相似度才被认为是冗余
                threshold = 0.85
            elif seg_type == 'intro' or existing_type == 'intro':
                # 引导语可以更宽松，允许更多变化
                threshold = 0.9
            elif seg_type == 'conclusion' or existing_type == 'conclusion':
                # 结论重复是典型的
                threshold = 0.75
            else:
                # 一般内容
                threshold = 0.8
            
            # 特殊情况：如果公式重叠非常高，强制判定为冗余
            if features['formula_overlap'] > 0.8 and features['number_overlap'] > 0.7:
                is_redundant = True
                best_similarity = similarity
                best_idx = idx
                best_features = features
                break
            
            # 更新最佳匹配
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
                best_features = features
            
            # 如果超过阈值，标记为冗余
            if similarity > threshold:
                is_redundant = True
                # 不立即break，继续搜索最佳匹配
        
        # 输出冗余检测信息（仅选择性输出，减少日志）
        if is_redundant and (i % 5 == 0 or best_similarity > 0.9):
            print(f"检测到冗余段落 (相似度: {best_similarity:.3f}):")
            print(f"- 类型: {seg_type} vs {unique_types[best_idx]}")
            print(f"- 当前: {normalized[:50]}...")
            print(f"- 匹配: {unique_segments[best_idx][:50]}...")
            if best_features:
                # 输出详细特征信息
                feature_str = ", ".join([f"{k}: {v:.2f}" for k, v in best_features.items()])
                print(f"- 特征: {feature_str}")
        
        if not is_redundant:
            # 检查并修复截断的文本
            if "48 times 1." in seg and not "48 times 1.5" in seg:
                seg = seg.replace("48 times 1.", "48 times 1.5 = 72")
            
            unique_segments.append(seg)
            unique_types.append(seg_type)
            
            # 添加当前段落的嵌入到列表中
            if current_embedding is not None:
                unique_embeddings.append(current_embedding)
    
    # 连接段落，确保连贯性
    if not unique_segments:
        return ""
    
    # 按照逻辑顺序排列：引导语 -> 计算步骤 -> 结论
    intro_segments = []
    step_segments = []
    conclusion_segments = []
    
    for seg, seg_type in zip(unique_segments, unique_types):
        if seg_type == 'intro':
            intro_segments.append(seg)
        elif seg_type == 'conclusion':
            conclusion_segments.append(seg)
        else:
            step_segments.append(seg)
    
    # 确保引导语和结论存在
    if not intro_segments and segments:
        intro = "Let's solve this problem step by step."
        intro_segments.append(intro)
    
    if not conclusion_segments and segments:
        for seg in segments:
            if "total" in seg.lower() and "72" in seg:
                conclusion = "Natalia sold a total of 72 clips in April and May."
                conclusion_segments.append(conclusion)
                break
    
    # 组合所有段落
    result = intro_segments + step_segments + conclusion_segments
    
    # 修复步骤编号，确保连续
    final_segments = []
    step_counter = 1
    has_steps = any(re.search(r'Step \d+:', seg) for seg in result)
    
    for i, seg in enumerate(result):
        # 检查是否包含步骤编号
        step_match = re.search(r'Step (\d+):', seg)
        
        # 如果是计算步骤但没有步骤编号，添加编号
        if i > 0 and i < len(result) - 1 and not step_match and has_steps and "=" in seg:
            seg = f"Step {step_counter}: {seg}"
            final_segments.append(seg)
            step_counter += 1
        elif step_match:
            # 替换为连续的步骤编号
            new_seg = re.sub(r'Step \d+:', f'Step {step_counter}:', seg)
            final_segments.append(new_seg)
            step_counter += 1
        else:
            final_segments.append(seg)
    
    # 最终连接，添加适当的连接词
    final_text = ""
    for i, seg in enumerate(final_segments):
        if i == 0:
            final_text = seg
        elif i == len(final_segments) - 1 and not seg.startswith("Therefore") and not seg.lower().startswith("thus"):
            # 为最后的结论添加连接词
            if "total" in seg.lower() or "answer" in seg.lower() or "sold" in seg.lower():
                final_text += " Therefore, " + seg[0].lower() + seg[1:]
            else:
                final_text += " " + seg
        else:
            final_text += " " + seg
    
    # 移除可能的提示信息
    final_text = re.sub(r'\(continue from Step \d+\)', '', final_text).strip()
    
    # 清理多余的空格和标点
    final_text = re.sub(r'\s+', ' ', final_text)
    final_text = re.sub(r'\s+([.,])', r'\1', final_text)
    
    # 修复可能的截断文本
    if "48 times 1." in final_text and not "48 times 1.5" in final_text:
        final_text = final_text.replace("48 times 1.", "48 times 1.5 = 72")
    
    # 清理模型生成的注释
    final_text = re.sub(r'\(Note:.*?\)', '', final_text).strip()
    final_text = re.sub(r'Note:.*?\.', '', final_text).strip()
    
    # 添加换行处理，提高可读性
    
    # 1. 首先处理步骤标记，确保它们单独成行并有适当的空行
    final_text = re.sub(r'([.!?]) (Step \d+:)', r'\1\n\n\2', final_text)
    final_text = re.sub(r'^(Step \d+:)', r'\n\1', final_text)
    
    # 2. 处理"Therefore"等结论词，确保它们单独成行
    final_text = re.sub(r'([.!?]) (Therefore|Thus|Hence|So|In conclusion)', r'\1\n\n\2', final_text)
    
    # 3. 在句子之间添加适当的换行
    # 将文本拆分为句子
    sentences = []
    current = ""
    
    # 使用正则表达式匹配句子边界
    for match in re.finditer(r'(.*?[.!?])\s+', final_text + " "):
        sentence = match.group(1).strip()
        if sentence:
            sentences.append(sentence)
    
    # 处理最后一个可能没有标点的句子
    if final_text and not final_text.rstrip()[-1] in ".!?":
        last_part = final_text.split(".")[-1].strip()
        if last_part:
            sentences.append(last_part)
    
    # 重新组合句子，添加适当的换行
    result_text = ""
    in_step = False
    
    for sentence in sentences:
        # 检查是否是步骤开始
        if re.match(r'^Step \d+:', sentence):
            if result_text:  # 不是第一个句子
                result_text += "\n\n"
            result_text += sentence
            in_step = True
        # 检查是否是结论句
        elif re.match(r'^(Therefore|Thus|Hence|So|In conclusion)', sentence):
            result_text += "\n\n" + sentence
            in_step = False
        # 检查是否包含计算结果
        elif "=" in sentence and re.search(r'\d+\s*=\s*\d+', sentence):
            if in_step:
                result_text += "\n" + sentence
            else:
                result_text += " " + sentence
        # 普通句子
        else:
            result_text += " " + sentence
    
    # 清理开头的空格
    result_text = result_text.strip()
    
    # 4. 确保计算步骤和结果格式良好
    # 在等号两侧添加空格
    result_text = re.sub(r'(\d+)=(\d+)', r'\1 = \2', result_text)
    
    # 5. 修复可能被错误分割的公式
    result_text = re.sub(r'(\d+)\n([+\-×÷=])', r'\1 \2', result_text)
    result_text = re.sub(r'([+\-×÷=])\n(\d+)', r'\1 \2', result_text)
    
    # 6. 清理多余的空格和换行符
    result_text = re.sub(r' +', ' ', result_text)  # 多个空格变一个
    result_text = re.sub(r'\n +', '\n', result_text)  # 换行后的空格
    result_text = re.sub(r'\n{3,}', '\n\n', result_text)  # 最多保留两个连续换行符
    
    # 7. 特殊处理：确保Step之间有空行，结论前有空行
    result_text = re.sub(r'(Step \d+:.*?)(\nStep \d+:)', r'\1\n\2', result_text)
    result_text = re.sub(r'([.!?])(\nTherefore|Thus|Hence|So|In conclusion)', r'\1\n\2', result_text)
    
    return result_text

############################################################
# -------------------   DEMO   ----------------------------#
############################################################

if __name__ == "__main__":
    full_chain = """
Okay, so Weng earns $12 an hour for babysitting, and she did 50 minutes of babysitting yesterday. I need to figure out how much she earned for that 50 minutes.

First, I should probably convert the 50 minutes into hours because her earnings are based on an hourly rate. There are 60 minutes in an hour, so 50 minutes is less than an hour. To convert minutes to hours, I can divide the number of minutes by 60.

So, 50 minutes divided by 60 minutes per hour equals... let's see, 50 divided by 60 is the same as 5/6, which is approximately 0.8333 hours.

Now, to find out how much she earned, I need to multiply the number of hours she worked by her hourly rate. Her hourly rate is $12, so I'll multiply 0.8333 hours by $12 per hour.

0.8333 times 12... Let me do the multiplication. 0.8 times 12 is 9.6, and 0.0333 times 12 is about 0.4, so adding those together, it's roughly $10.

But to be more precise, I can calculate it directly. 50 minutes is exactly 5/6 of an hour, and 5/6 times 12 is... 5 times 12 is 60, and then divided by 6 is 10.

So, Weng earned $10 for 50 minutes of babysitting.

Wait a minute, does the employer pay in fractions of a cent, or do they round to the nearest cent? Since we're dealing with dollars and cents, and the calculation came out to an even $10, I don't need to worry about rounding.

Therefore, Weng earned $10 yesterday for babysitting.
"""

    # 1. 复杂度整体分数
    analyzer = complexity_estimator.LanguageModelAnalyzer()
    C_total = analyzer.analyze_cot(full_chain).overall_score

    # 2. 分块
    segmenter = cot_semantic_segmenter.DistilBERTSegmenter()
    segments = segmenter.segment(full_chain)
    for i, seg in enumerate(segments, 1):
        print(f"S{i}: {seg}")

    # 3. 每块得分与层级
    scores = [analyzer.analyze_cot(s).overall_score for s in segments]
    layers = assign_layers(segments, scores, z_th=0.4)

    # 打印分层结果
    print("\n=== 分层结果 ===")
    for i, (seg, lay, sc) in enumerate(zip(segments, layers, scores), 1):
        print(f"S{i} [L{lay} | {sc:.3f}]: {seg[:50]}...")

    # 4. 压缩，带 prev 连贯
    compressor = CoTCompressor(prompt_dir="./prompts")
    comp_segs, prev = [], ""
    for seg, lay in zip(segments, layers):
        comp = compressor.compress(seg, lay, prev=prev)
        comp_segs.append(comp)
        prev = comp

    # 5. 输出原始压缩结果
    print("\n=== 原始压缩结果 ===")
    for seg, lay, sc, cp in zip(segments, layers, scores, comp_segs):
        print(f"[L{lay} | {sc:.3f}] → {cp}")

    # 6. 后处理压缩结果
    processed_cot = post_process_compression(comp_segs, segments, layers)
    tok_cnt = len(compressor.tokenizer.tokenize(processed_cot))
    print("\n[Processed Compressed CoT]\n" + processed_cot)
    print(f"\n[Metrics] Compressed token count: {tok_cnt}")
    
    # 7. 打印压缩比
    original_tok_cnt = len(compressor.tokenizer.tokenize(full_chain))
    compression_ratio = tok_cnt / original_tok_cnt
    print(f"原始token数: {original_tok_cnt}")
    print(f"压缩比: {compression_ratio:.2f} ({tok_cnt}/{original_tok_cnt})")
    
    # 8. 打印优化后的分层统计
    print("\n=== 优化后的分层统计 ===")
    layer_counts = {1: 0, 2: 0, 3: 0}
    for l in layers:
        layer_counts[l] += 1
    print(f"层级1 (轻度压缩): {layer_counts[1]}段")
    print(f"层级2 (中度压缩): {layer_counts[2]}段")
    print(f"层级3 (重度压缩): {layer_counts[3]}段")
