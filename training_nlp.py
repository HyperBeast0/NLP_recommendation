from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import json


# 1. Подготовка данных
def preprocess_data(data):
    tokenizer = BertTokenizerFast.from_pretrained("DeepPavlov/rubert-base-cased")

    tokenized_data = []
    for item in data:
        text = item[0]  # Текст находится в первом элементе списка
        entities = item[1]["entities"] if len(item) > 1 and "entities" in item[1] else []

        # Токенизация текста
        tokens = tokenizer(text, truncation=True, padding="max_length",
                           max_length=128, return_offsets_mapping=True)
        labels = [0] * len(tokens['input_ids'])

        # Проставляем метки для сущностей
        for start, end, label in entities:
            for idx, (offset_start, offset_end) in enumerate(tokens['offset_mapping']):
                if offset_start >= start and offset_end <= end:
                    labels[idx] = label_to_id.get(label, 0)

        tokenized_data.append({
            "input_ids": tokens['input_ids'],
            "attention_mask": tokens['attention_mask'],
            "labels": labels
        })

    return Dataset.from_list(tokenized_data)


# Загрузка данных из JSON файла
with open("Data/train_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("Data/validation_data.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# Словарь для преобразования меток в ID
label_to_id = {
    "O": 0,
    "PERSON_F": 1,
    "PERSON_M": 2,
    "EVENT": 3,
    "OBJECT": 4
}
id_to_label = {v: k for k, v in label_to_id.items()}

# Преобразуем данные
dataset = preprocess_data(data)
eval_dataset = preprocess_data(eval_data)

# 2. Настройка модели
name_model = "DeepPavlov/rubert-base-cased"
model = BertForTokenClassification.from_pretrained(name_model, num_labels=len(label_to_id))

# 3. Обучение
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-5,
    per_device_train_batch_size=25,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=10,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    tokenizer=BertTokenizerFast.from_pretrained("DeepPavlov/rubert-base-cased")
)

# Запуск обучения
trainer.train()

# Сохранение модели
model.save_pretrained("./my_rubert_model")

