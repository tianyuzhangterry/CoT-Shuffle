from complexity_estimator import LanguageModelAnalyzer, _SENT_RE

# 初始化分析器
analyzer = LanguageModelAnalyzer()

# 测试不同类型的数学推理
cot_examples = [
    # 示例0 - 复杂积分
    """
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
    """,
    
    # 示例1 - 微分
    """
    I need to find the derivative of f(x) = x³ - 4x² + 5x - 7 with respect to x.
    
    To find the derivative, I'll apply the power rule: d/dx[x^n] = n·x^(n-1)
    
    For f(x) = x³ - 4x² + 5x - 7:
    
    The derivative of x³ is 3x²
    The derivative of -4x² is -4·2x = -8x
    The derivative of 5x is 5
    The derivative of -7 is 0 (since constants differentiate to zero)
    
    Adding these terms:
    f'(x) = 3x² - 8x + 5
    
    Therefore, the derivative of f(x) = x³ - 4x² + 5x - 7 is f'(x) = 3x² - 8x + 5.
    """,
    
    # 示例2 - 简单方程求解
    """
    I need to solve the equation 2x + 5 = 13.
    
    First, I'll subtract 5 from both sides to isolate the term with x.
    2x + 5 - 5 = 13 - 5
    2x = 8
    
    Now I'll divide both sides by 2 to solve for x.
    2x/2 = 8/2
    x = 4
    
    To verify this solution, I'll substitute x = 4 back into the original equation.
    2(4) + 5 = 8 + 5 = 13 ✓
    
    Therefore, the solution to the equation 2x + 5 = 13 is x = 4.
    """,
    
    # 示例3 - 无理数证明
    """
    To prove that √2 is irrational, I'll use proof by contradiction.
    
    Assume √2 is rational, so √2 = p/q where p,q are integers with gcd(p,q) = 1.
    
    Then 2 = p²/q², so 2q² = p².
    
    This means p² is even, therefore p is even.
    
    Let p = 2k for some integer k.
    
    Substituting: 2q² = (2k)² = 4k², so q² = 2k².
    
    This means q² is even, therefore q is even.
    
    But if both p and q are even, then gcd(p,q) ≥ 2, contradicting our assumption.
    
    Therefore, √2 must be irrational.
    """,
    
    # 示例4 - 非数学推理
    """
    The Treaty of Versailles was signed in 1919, marking the end of World War I. 
    It imposed harsh penalties on Germany, including territorial losses, military restrictions, 
    and massive reparation payments. Many historians argue that these punitive measures 
    contributed to economic hardship and political instability in Germany during the 1920s and 1930s, 
    creating conditions that facilitated the rise of extremist movements like the Nazi Party. 
    The treaty's legacy demonstrates how peace settlements that are perceived as unjust 
    can sometimes sow the seeds for future conflicts.
    """
]

# 标签
labels = ["复杂积分", "微分", "方程求解", "无理数证明", "历史分析"]

print("=== 单个CoT分析 ===")
result = analyzer.analyze_cot(cot_examples[3])  # 无理数证明
print(result)
print()

print("=== 比较不同类型CoT ===")
results = analyzer.compare_cots(cot_examples, labels)

# 打印结果
for item in results['results']:
    print(f"{item['label']} ({item['length']} 词, {item['sentences']} 句):")
    print(f"  总体得分: {item['score'].overall_score:.3f}")
    print(f"  推理深度: {item['score'].reasoning_depth:.3f}")
    print(f"  概念密度: {item['score'].concept_density:.3f}")
    print(f"  单步复杂度: {item['score'].step_complexity:.3f}\n")

# 打印统计信息
print("=== 统计信息 ===")
for k, v in results['statistics'].items():
    print(f"{k}: {v:.3f}") 