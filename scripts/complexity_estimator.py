import re, math, numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer, AutoModel

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_LOGIC = {"if", "then", "thus", "therefore", "hence", "so", "because", "since", "as", "follows", "conclude", "proves", "implies", "consequently", "given", "assume", "let", "suppose"}
_ADVANCED_MATH = {"integral", "derivative", "differentiate", "integrate", "calculus", "theorem", "lemma", "proof", "prove", "function", "equation", "formula", "expression", "variable", "constant", "coefficient", "polynomial", "exponential", "logarithm", "trigonometric", "sine", "cosine", "tangent", "vector", "matrix"}

@dataclass
class CoTComplexityScore:
    """CoT复杂度分析结果 - 专门针对推理链优化"""
    overall_score: float
    reasoning_depth: float      # 推理深度
    concept_density: float      # 概念密度
    step_complexity: float      # 单步复杂度
    word_count: int            # 词数
    sentence_count: int        # 句数
    
    def __str__(self):
        return (f"({self.word_count} 词, {self.sentence_count} 句):\n"
                f"  总体得分: {self.overall_score:.3f} | "
                f"推理深度: {self.reasoning_depth:.3f} | "
                f"概念密度: {self.concept_density:.3f} | "
                f"单步复杂度: {self.step_complexity:.3f}")

class LanguageModelAnalyzer:
    """使用预训练语言模型的复杂度分析器"""
    
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        初始化分析器
        
        Args:
            model_name: 使用的预训练模型名称，默认使用DistilBERT模型
        """
        print(f"正在加载模型 {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # 检测设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"模型已加载到设备: {self.device}")
    
    def _prepare_input(self, text: str) -> torch.Tensor:
        """准备模型输入"""
        # DistilBERT有512个token的限制，所以需要截断长文本
        encoded_input = self.tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        
        # 将输入移动到正确的设备
        input_ids = encoded_input["input_ids"].to(self.device)
        attention_mask = encoded_input["attention_mask"].to(self.device)
        
        return input_ids, attention_mask
    
    def _compute_embedding_complexity(self, text: str) -> float:
        """计算文本嵌入的复杂度"""
        input_ids, attention_mask = self._prepare_input(text)
        
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 获取最后一层的隐藏状态 (处理不同的输出格式)
            if isinstance(outputs, tuple):
                last_hidden_state = outputs[0]  # 元组格式
            else:
                last_hidden_state = outputs.last_hidden_state  # 对象格式
            
            # 使用注意力掩码来获取有效token的平均表示
            # 首先，将注意力掩码扩展为与隐藏状态相同的形状
            attention_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            
            # 使用掩码计算有效token的和
            sum_hidden = torch.sum(last_hidden_state * attention_expanded, 1)
            # 计算有效token的数量
            sum_mask = torch.sum(attention_mask, 1).unsqueeze(-1)
            # 计算平均表示
            avg_hidden = sum_hidden / sum_mask
            
            # 计算嵌入向量的L2范数作为复杂度指标
            embedding_norm = torch.norm(avg_hidden, dim=1).item()
            
            # 调整归一化方式，使用线性缩放加上sigmoid函数
            normalized_complexity = self._enhanced_normalize(embedding_norm)
            
            return normalized_complexity
    
    def _enhanced_normalize(self, value):
        """增强的归一化函数，结合线性缩放和sigmoid函数"""
        # 线性缩放到合适范围
        scaled = value / 15.0
        
        # 应用sigmoid函数增强差异
        sigmoid = 1.0 / (1.0 + math.exp(-5.0 * (scaled - 0.5)))
        
        # 缩放到0.3-0.8的范围，保持与原始评分相似的数值范围
        adjusted = 0.3 + sigmoid * 0.5
        
        return adjusted
    
    def _compute_reasoning_depth(self, text: str, sentences: List[str]) -> float:
        """
        计算推理深度 - 使用语言模型评估
        """
        if not sentences:
            return 0.0
            
        # 获取整个文本的嵌入复杂度
        text_complexity = self._compute_embedding_complexity(text)
        
        # 考虑句子数量的对数因子，调整权重增加差异
        sentence_factor = min(0.9, math.log(len(sentences) + 1) / math.log(8))
        
        # 结合两个因素，增加文本复杂度的权重
        depth_score = text_complexity * 0.7 + sentence_factor * 0.3
        
        # 确保分数在合理范围内
        return min(depth_score, 0.95)
    
    def _compute_concept_density(self, text: str, sentences: List[str]) -> float:
        """
        计算概念密度 - 使用模型评估句子的语义丰富度
        """
        if not sentences:
            return 0.0
        
        # 对于概念密度，我们使用模型对每个句子单独评分，然后计算加权平均值
        sentence_scores = []
        weights = []
        
        for sentence in sentences:
            if len(sentence.split()) < 3:  # 跳过过短的句子
                continue
                
            # 计算句子的嵌入复杂度
            sentence_complexity = self._compute_embedding_complexity(sentence)
            sentence_scores.append(sentence_complexity)
            
            # 句子长度作为权重
            weights.append(min(len(sentence.split()) / 10, 1.0))
        
        # 计算加权平均概念密度
        if sentence_scores:
            concept_density = sum(s * w for s, w in zip(sentence_scores, weights)) / sum(weights)
        else:
            concept_density = 0.4  # 默认中等密度
            
        return min(concept_density, 0.95)
    
    def _compute_step_complexity(self, sentences: List[str]) -> float:
        """
        计算平均每步的复杂度 - 使用模型评估每个句子的复杂度
        """
        if not sentences:
            return 0.0
        
        # 使用模型计算每个句子的复杂度
        step_scores = []
        for sentence in sentences:
            if len(sentence.split()) < 3:  # 跳过过短的句子
                continue
                
            # 计算句子的嵌入复杂度
            sentence_complexity = self._compute_embedding_complexity(sentence)
            step_scores.append(sentence_complexity)
        
        # 返回平均单步复杂度
        return np.mean(step_scores) if step_scores else 0.4
    
    def analyze_cot(self, cot_text: str, normalize_length: bool = True) -> CoTComplexityScore:
        """
        分析CoT的复杂度
        
        Args:
            cot_text: CoT推理链文本
            normalize_length: 是否进行长度归一化（推荐True）
        """
        # 预处理
        sentences = [s.strip() for s in _SENT_RE.split(cot_text.strip()) if s.strip()]
        if not sentences:
            return CoTComplexityScore(0, 0, 0, 0, 0, 0)
        
        n_words = len(cot_text.split())
        
        # 计算各项指标
        reasoning_depth = self._compute_reasoning_depth(cot_text, sentences)
        concept_density = self._compute_concept_density(cot_text, sentences)
        step_complexity = self._compute_step_complexity(sentences)
        
        # 调整权重分配
        if normalize_length:
            weights = {
                'reasoning_depth': 0.45,   # 推理深度权重
                'concept_density': 0.35,   # 概念密度
                'step_complexity': 0.20    # 单步复杂度
            }
        else:
            # 不归一化时更重视总体指标
            weights = {
                'reasoning_depth': 0.50,
                'concept_density': 0.35,
                'step_complexity': 0.15
            }
        
        # 计算总分
        overall_score = (
            reasoning_depth * weights['reasoning_depth'] +
            concept_density * weights['concept_density'] +
            step_complexity * weights['step_complexity']
        )
        
        # 确保分数在合理范围内
        overall_score = max(0.05, min(overall_score, 0.95))
        
        return CoTComplexityScore(
            overall_score=round(overall_score, 3),
            reasoning_depth=round(reasoning_depth, 3),
            concept_density=round(concept_density, 3),
            step_complexity=round(step_complexity, 3),
            word_count=n_words,
            sentence_count=len(sentences)
        )
    
    def compare_cots(self, cot_list: List[str], labels: Optional[List[str]] = None) -> Dict:
        """
        比较多个CoT的复杂度
        """
        results = []
        labels = labels or [f"CoT_{i+1}" for i in range(len(cot_list))]
        
        for i, cot in enumerate(cot_list):
            score = self.analyze_cot(cot)
            results.append({
                'label': labels[i],
                'score': score,
                'length': len(cot.split()),
                'sentences': len([s for s in _SENT_RE.split(cot.strip()) if s.strip()])
            })
        
        # 统计信息
        overall_scores = [r['score'].overall_score for r in results]
        score_std = np.std(overall_scores)
        score_range = max(overall_scores) - min(overall_scores)
        
        return {
            'results': results,
            'statistics': {
                'mean_score': np.mean(overall_scores),
                'std_deviation': score_std,
                'score_range': score_range,
                'consistency': 1.0 - min(score_std * 2, 1.0)  # 一致性指标
            }
        }

# 使用示例
if __name__ == "__main__":
    # 检测CUDA是否可用
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    else:
        print("CUDA不可用")
    
    # 默认使用GPU（如果可用）
    analyzer = LanguageModelAnalyzer()
    
    # 示例CoT文本
    sample_cot = """
    We want to evaluate the improper integral
    I = ∫₀^∞ x³ / (eˣ − 1) dx
    in closed form. A glance shows it resembles Bose–Einstein integrals that connect the Gamma and Riemann-zeta functions. In fact, one recalls the general identity that for any real α > 1,
    ∫₀^∞ x^(α−1) / (eˣ − 1) dx = Γ(α)·ζ(α),
    which is proved by expanding 1∕(eˣ − 1) = Σ_{n=1}^∞ e^(−nx), swapping summation and integration under uniform convergence, then noting
    ∫₀^∞ x^(α−1)e^(−nx) dx = Γ(α)·n^(−α),
    so that summing over n gives Γ(α)ζ(α).
    In our case α = 4 since the numerator is x³, so
    I = Γ(4)·ζ(4).
    We have Γ(4) = 3! = 6, and the Basel-type result ζ(4) = π⁴∕90 is taken as known. Hence
    I = 6 × (π⁴∕90) = π⁴∕15.
    """
    
    print("=== CoT复杂度分析 ===")
    result = analyzer.analyze_cot(sample_cot)
    print(result)