"""
基于选用的基准数据集构建任务感知模型（BERT/DeBERTa 等）的训练数据。

数据集来源：
- CommonsenseQA: https://huggingface.co/datasets/commonsense_qa
- BBH (Big-Bench Hard): https://huggingface.co/datasets/lukaemon/bbh
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
"""

import os
import random
from datasets import load_dataset, get_dataset_config_names, Dataset


def build_mixed_perception_dataset(
        cqa_samples=5000,
        gsm8k_samples=2500,
        bbh_samples=2500,
        cache_dir="D:\HuggingFace",
        output_dir="./bert_train_data"
):
    """
    构建用于训练任务感知模块的混合数据集，包含 CommonsenseQA、GSM8K 以及所有 BBH 子类。
    """
    print("========构建混合数据集========")

    # 确保缓存目录存在，避免每次都从头下载
    os.makedirs(cache_dir, exist_ok=True)
    # 处理 CommonsenseQA（标签 0：简单任务）

    print("\n========[1/3] 正在处理 CommonsenseQA (Label 0)========")
    cqa = load_dataset("commonsense_qa", split="train", cache_dir=cache_dir)
    cqa_texts = []
    for item in cqa:
        choices_text = ", ".join(item["choices"]["text"])
        text = f"Question: {item['question']}\nChoices: {choices_text}"
        cqa_texts.append(text)

    cqa_sampled = random.sample(cqa_texts, min(cqa_samples, len(cqa_texts)))
    data_0 = [{"text": text, "label": 0} for text in cqa_sampled]
    print(f"成功提取 CommonsenseQA 数据: {len(data_0)} 条")

    # 处理 GSM8K（标签 1：复杂任务）
    print("\n========[2/3] 正在处理 GSM8K (Label 1)========")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train", cache_dir=cache_dir)
    gsm8k_texts = [f"Question: {item['question']}" for item in gsm8k]

    gsm8k_sampled = random.sample(gsm8k_texts, min(gsm8k_samples, len(gsm8k_texts)))
    data_1_gsm8k = [{"text": text, "label": 1} for text in gsm8k_sampled]
    print(f"✅ 成功提取 GSM8K 数据: {len(data_1_gsm8k)} 条")


    # 3. 处理 BBH（标签 1：复杂任务）- 遍历所有子类
    print("\n========[3/3] 正在处理 BBH 全子类 (Label 1)========")
    # 获取 BBH 的所有子集名称（共 20+ 个）
    bbh_configs = get_dataset_config_names("lukaemon/bbh")
    bbh_all_texts = []

    for config in bbh_configs:
        try:
            # BBH 通常只有测试划分（test）
            bbh_subset = load_dataset("lukaemon/bbh", config, split="test", cache_dir=cache_dir)
            bbh_all_texts.extend([f"Question: {item['input']}" for item in bbh_subset])
        except Exception as e:
            print(f"跳过子集 {config}，原因: {e}")

    print(f"   -> 从所有 BBH 子集中共汇总了 {len(bbh_all_texts)} 条原始数据。")
    # 从拼接后的全集中抽取指定数量
    bbh_sampled = random.sample(bbh_all_texts, min(bbh_samples, len(bbh_all_texts)))
    data_1_bbh = [{"text": text, "label": 1} for text in bbh_sampled]
    print(f"✅ 成功抽取 BBH 数据: {len(data_1_bbh)} 条")

    # 4并、打乱并保存至本地
    print("\n========正在合并并打乱数据========")
    final_data_list = data_0 + data_1_gsm8k + data_1_bbh
    random.shuffle(final_data_list)

    # 转换为 Hugging Face Dataset 对象
    final_dataset = Dataset.from_list(final_data_list)

    # 方式 A：保存为 Hugging Face 磁盘格式
    final_dataset.save_to_disk(output_dir)
    print(f"数据已存为 HF 格式至: {output_dir}")

    # 方式 B：顺便存一份 JSONL 格式，方便人工审阅数据
    # jsonl_path = f"{output_dir}/human_readable.jsonl"
    # with open(jsonl_path, "w", encoding="utf-8") as f:
    #     for item in final_data_list:
    #         f.write(json.dumps(item, ensure_ascii=False) + "\n")
    # print(f"人类可读的 JSONL 数据已保存至: {jsonl_path}")

    print(
        f"\n最终数据集总样本数: {len(final_dataset)} (Label 0: {len(data_0)}, Label 1: {len(data_1_gsm8k) + len(data_1_bbh)})")

    return final_dataset


if __name__ == "__main__":
    # 执行构建脚本
    dataset = build_mixed_perception_dataset()
