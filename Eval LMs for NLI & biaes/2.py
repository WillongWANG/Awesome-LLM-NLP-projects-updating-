import json
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

# 1. 加载数据
with open('dataset.json') as f:
    dataset = json.load(f)

# 2. 加载预训练NLI模型
model_name = "roberta-large-mnli" # textattack/bert-base-uncased-MNLI
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config.max_position_embeddings = 1157 # for roberta-large-mnli
model = AutoModelForSequenceClassification.from_config(config)
print(model.config.max_position_embeddings)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 3. 定义标签映射
label_map = {
    "major_inaccurate": 1,
    "minor_inaccurate": 1,
    "accurate": 0
}

# 4. 处理数据和预测
all_true = []
all_pred = []
token_count=[]

for item in tqdm(dataset, desc="Processing items"): #238项
    premise = item["wiki_bio_text"]
    hypotheses = item["gpt3_sentences"]
    annotations = item["annotation"]
    
    for hyp, ann in zip(hypotheses, annotations):
        tokens = tokenizer(premise, hyp, truncation=False)["input_ids"]
        token_count.append(len(tokens))
        # max(token_count)=1093 for bert
        # max(token_count)=1155 for roberta
        # 构建模型输入
        inputs = tokenizer(
            premise, 
            hyp,
            return_tensors="pt",
        ).to(device)
        
        # 模型推理
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取预测结果
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        
        # MNLI标签映射: 0=entailment, 1=neutral, 2=contradiction
        # 将entailment视为准确，其他视为不准确
        pred = 0 if predicted_label == 0 else 1
        
        # 转换真实标签
        true_label = label_map[ann]
        
        all_true.append(true_label)
        all_pred.append(pred)

# 5. 计算指标
print("Accuracy:", accuracy_score(all_true, all_pred))
print("Precision:", precision_score(all_true, all_pred, average=None)) # 对应true_label 0，1
print("Recall:", recall_score(all_true, all_pred, average=None))
print("F1:", f1_score(all_true, all_pred, average=None))
print("\nClassification Report:")
print(classification_report(all_true, all_pred))

# textattack/bert-base-uncased-MNLI:
# Accuracy: 0.4921383647798742
# Precision: [0.27728614 0.73737374]
# Recall: [0.54651163 0.47198276]
# F1: [0.36790607 0.57555848]
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.28      0.55      0.37       516
#            1       0.74      0.47      0.58      1392

#     accuracy                           0.49      1908
#    macro avg       0.51      0.51      0.47      1908
# weighted avg       0.61      0.49      0.52      1908

# roberta-large-mnli
# Accuracy: 0.6184486373165619
# Precision: [0.27056277 0.72959889]
# Recall: [0.24224806 0.7579023 ]
# F1: [0.25562372 0.74348132]

# Classification Report:
#               precision    recall  f1-score   support

#            0       0.27      0.24      0.26       516
#            1       0.73      0.76      0.74      1392

#     accuracy                           0.62      1908
#    macro avg       0.50      0.50      0.50      1908
# weighted avg       0.61      0.62      0.61      1908