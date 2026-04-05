import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType


class TaskPerceptionBERT:
    def __init__(self, model_name: str = "microsoft/deberta-v3-small", use_lora: bool = True):
        """
        初始化任务感知模型
        默认使用轻量且在推理分类任务上极强的 DeBERTa-v3-small 英文基座
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 加载用于序列分类的模型，设定 num_labels=2 (0: 单模型/小模型, 1: 多Agent/大模型)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        if use_lora:
            # 引入 LoRA 进行参数高效微调
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query_proj", "value_proj"]  # 针对注意力机制模块注入
            )
            self.model = get_peft_model(self.model, peft_config)
            print("\nLoRA 可训练参数比例:")
            self.model.print_trainable_parameters()

        self.model.to(self.device)

    def train_from_local(self, local_data_path: str, output_dir: str = "./task_perception_lora_model"):
        """
        直接从本地加载合并好的 Dataset，自动划分验证集，并应用动态 Padding 进行微调
        """
        print(f"\n 正在从本地极速加载数据集: {local_data_path}")
        dataset = load_from_disk(local_data_path)

        # 自动划分 80% 训练集，20% 验证集
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        print(f"数据集划分完成：训练集 {len(train_dataset)} 条，验证集 {len(eval_dataset)} 条。")

        def tokenize_function(examples):
            # 不使用全局 padding，仅截断，利用 DataCollator 实现 Batch 维度的动态 Padding
            return self.tokenizer(examples["text"], truncation=True, max_length=256)

        print("\n 正在进行 Tokenize...")
        tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # 动态 Padding 收集器，大幅降低无效显存占用和计算量
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        print("\n配置训练参数并启动微调...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=3e-4,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=3,
            weight_decay=0.01,
            eval_strategy="epoch",  # 每个 epoch 结束验证一次
            save_strategy="epoch",
            load_best_model_at_end=True,  # 训练结束后恢复最优验证集 Loss 的权重
            metric_for_best_model="loss",
            logging_steps=50,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
        )

        trainer.train()

        print("\n训练完成，正在保存模型...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"任务感知模块已成功保存至: {output_dir}")

    def predict(self, query: str) -> dict:
        """
        推理函数，输出分类结果
        :return: dict 包含 complexity_level (0或1) 和 confidence
        """
        self.model.eval()
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
            probability = torch.softmax(logits, dim=-1)[0][predicted_class_id].item()

        return {
            "complexity_level": predicted_class_id,
            "confidence": round(probability, 4)
        }


if __name__ == "__main__":
    perception_module = TaskPerceptionBERT()
    perception_module.train_from_local(local_data_path="../Dataset/bert_train_data")

    # 3. 模拟接收系统的 Query 进行路由判定
    test_query_easy = "Question: The sun rises in which direction?\nChoices: East, West, North, South"
    test_query_hard = "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"

    print("\n[测试] 简单任务预测:", perception_module.predict(test_query_easy))
    print("[测试] 复杂任务预测:", perception_module.predict(test_query_hard))