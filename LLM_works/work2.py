# 导入所需库
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
import datasets
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 加载数据集
data_files = {
    "train": "reviews.csv",
    "test": "reviews.csv"
}
raw_datasets = datasets.load_dataset("csv", data_files=data_files, delimiter=",")

# 模型与分词器加载
model_name_or_path = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=5)


# 定义数据预处理函数
def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True)


# 对数据集进行分词处理
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 查看预处理后的数据
print(tokenized_datasets)

# 准备训练和评估数据集
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(50))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(50))


# 定义评估指标计算函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# 定义训练参数
training_args = TrainingArguments(
    output_dir='reviews_trainer',
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    num_train_epochs=3,
    warmup_ratio=0.2,
    logging_dir='./reviews_train_logs',
    logging_strategy="epoch",
    save_strategy="epoch",
    report_to="tensorboard"
)

# 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 开始训练并保存模型
trainer.train()
trainer.save_model()
