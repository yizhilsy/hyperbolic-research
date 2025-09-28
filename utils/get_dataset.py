from datasets import load_dataset
import os
import glob

BASE_DIR = "/home/lsy/shared_data/PKU-Alignment/MM-SafetyBench/data"

def get_dataset(task_name: str):
    """
    根据任务名（如 "EconomicHarm", "Fraud" 等）加载对应数据集目录下的所有 parquet 文件
    返回 DatasetDict，可以通过 dataset["SD"], dataset["TYPO"] 访问
    """
    task_dir = os.path.join(BASE_DIR, task_name)
    if not os.path.exists(task_dir):
        raise ValueError(f"任务 {task_name} 的目录不存在: {task_dir}")
    
    # 收集该目录下的所有 parquet 文件
    parquet_files = glob.glob(os.path.join(task_dir, "*.parquet"))
    if not parquet_files:
        raise ValueError(f"在 {task_dir} 下没有找到 parquet 文件")
    
    # 构造 data_files dict: {"SD": ".../SD.parquet", ...}
    data_files = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in parquet_files
    }
    
    dataset = load_dataset("parquet", data_files=data_files)
    return dataset
