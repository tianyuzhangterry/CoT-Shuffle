import pandas as pd
import os
import torch
from transformers import AutoTokenizer
import re
import sys

# 导入自定义模块
import cot_compressor
import cot_semantic_segmenter
import complexity_estimator

def generate_compressed_dataset(batch_size=50, max_samples=None):
    """
    加载处理后的MixChain-Z-GSM8K数据集，添加压缩链和压缩比率。
    使用批处理方式避免内存溢出。
    
    参数:
        batch_size: 每批处理的样本数量
        max_samples: 最大处理样本数，None表示处理全部
    """
    print("开始处理压缩数据集...")
    
    # 1. 检查输入文件
    input_path = os.path.join("processed_data", "mixchain_gsm8k_train.csv")
    if not os.path.exists(input_path):
        print(f"错误：找不到输入文件 {input_path}，请先运行 process_dataset.py")
        return
    
    # 2. 初始化压缩器和分析器
    print("初始化压缩器和分析器...")
    try:
        compressor = cot_compressor.CoTCompressor(prompt_dir="./prompts")
        analyzer = complexity_estimator.LanguageModelAnalyzer()
        segmenter = cot_semantic_segmenter.DistilBERTSegmenter()
        tokenizer = compressor.tokenizer
        print("初始化成功")
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 3. 创建输出目录
    output_path = os.path.join("processed_data", "mixchain_gsm8k_train_compressed.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 4. 获取数据集总行数
    total_rows = sum(1 for _ in pd.read_csv(input_path, chunksize=1))
    if max_samples is not None:
        total_rows = min(total_rows, max_samples)
    print(f"数据集总行数: {total_rows}")
    
    # 5. 批处理变量初始化
    processed_count = 0
    successful_count = 0
    total_orig_tokens = 0
    total_comp_tokens = 0
    
    # 6. 准备结果文件头
    result_columns = ['question', 'answer', 'target', 'solution_1', 'solution_1_token', 
                     'compressed_chain', 'original_tokens', 'compressed_tokens', 'compression_ratio']
    
    # 写入CSV头
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(','.join(result_columns) + '\n')
    
    # 7. 分批处理数据
    print("开始分批处理数据...")
    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=batch_size)):
        if max_samples is not None and processed_count >= max_samples:
            break
            
        print(f"\n处理批次 {chunk_idx+1}, 行 {processed_count+1}-{min(processed_count+len(chunk), total_rows)}/{total_rows}")
        
        # 初始化新列
        chunk['compressed_chain'] = ""
        chunk['original_tokens'] = 0
        chunk['compressed_tokens'] = 0
        chunk['compression_ratio'] = 0.0
        
        # 处理当前批次
        batch_success = 0
        for i, row in chunk.iterrows():
            if processed_count % 10 == 0:
                print(f"处理第 {processed_count+1}/{total_rows} 条记录...")
                
            # 获取原始解题过程
            original_solution = row['solution_1']
            if pd.isna(original_solution) or not isinstance(original_solution, str):
                processed_count += 1
                continue
                
            # 测量原始token数
            original_tokens = len(tokenizer.tokenize(original_solution))
            
            try:
                # 1. 分段
                segments = segmenter.segment(original_solution)
                
                # 2. 每个段落得分与层级
                scores = [analyzer.analyze_cot(s).overall_score for s in segments]
                layers = cot_compressor.assign_layers(segments, scores)
                
                # 3. 生成压缩段落
                comp_segs, prev = [], ""
                for seg, lay in zip(segments, layers):
                    comp = compressor.compress(seg, lay, prev=prev)
                    comp_segs.append(comp)
                    prev = comp
                    
                # 4. 后处理压缩结果
                processed_cot = cot_compressor.post_process_compression(comp_segs, segments, layers)
                
                # 5. 测量压缩后token数
                compressed_tokens = len(tokenizer.tokenize(processed_cot))
                
                # 6. 计算压缩比
                compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0
                
                # 更新数据
                chunk.at[i, 'compressed_chain'] = processed_cot
                chunk.at[i, 'original_tokens'] = original_tokens
                chunk.at[i, 'compressed_tokens'] = compressed_tokens
                chunk.at[i, 'compression_ratio'] = compression_ratio
                
                # 更新统计信息
                batch_success += 1
                total_orig_tokens += original_tokens
                total_comp_tokens += compressed_tokens
                
            except Exception as e:
                print(f"处理记录 {processed_count+1} 时出错: {e}")
            
            processed_count += 1
            if max_samples is not None and processed_count >= max_samples:
                break
        
        # 将当前批次追加到结果文件
        chunk.to_csv(output_path, mode='a', header=False, index=False)
        successful_count += batch_success
        
        # 显示当前批次统计
        print(f"当前批次成功处理: {batch_success}/{len(chunk)}")
        
        # 释放内存
        del chunk
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 8. 显示最终统计
    avg_ratio = total_comp_tokens / total_orig_tokens if total_orig_tokens > 0 else 0
    print(f"\n处理完成! 总共处理: {processed_count} 条记录")
    print(f"成功压缩: {successful_count}/{processed_count} 条记录")
    print(f"平均压缩比: {avg_ratio:.4f} ({total_comp_tokens}/{total_orig_tokens})")
    print(f"已保存压缩数据集到 {output_path}")
    
def main():
    # 可以在这里调整批大小和最大样本数
    generate_compressed_dataset(batch_size=50, max_samples=100)  # 设置为None处理全部数据

if __name__ == "__main__":
    main() 