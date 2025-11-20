from datasets import load_dataset
import pandas as pd
import os

def process_mixchain_dataset():
    """
    加载并处理MixChain-Z-GSM8K数据集，只保留指定字段
    """
    print("正在加载MixChain-Z-GSM8K数据集...")
    ds = load_dataset("horseee/MixChain-Z-GSM8K")
    
    # 查看数据集结构
    print(f"数据集结构: {ds}")
    
    # 处理每个分割
    for split in ds.keys():
        print(f"\n处理 {split} 分割...")
        
        # 转换为DataFrame以便于处理
        df = pd.DataFrame(ds[split])
        
        # 只保留指定字段
        keep_columns = ['question', 'answer', 'target', 'solution_1', 'solution_1_token']
        available_columns = [col for col in keep_columns if col in df.columns]
        missing_columns = set(keep_columns) - set(available_columns)
        
        if missing_columns:
            print(f"警告: 以下字段在数据集中不存在: {missing_columns}")
        
        df_filtered = df[available_columns]
        
        # 显示数据集信息
        print(f"{split} 分割大小: {len(df_filtered)} 条")
        print(f"保留的字段: {available_columns}")
        print(f"前3条数据示例:")
        print(df_filtered.head(3))
        
        # 保存处理后的数据集
        output_dir = "processed_data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"mixchain_gsm8k_{split}.csv")
        df_filtered.to_csv(output_path, index=False)
        print(f"已保存到 {output_path}")

def main():
    """主函数"""
    process_mixchain_dataset()

if __name__ == "__main__":
    main() 