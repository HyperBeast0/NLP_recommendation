import random
import json
import pandas as pd
import pymorphy2
import re

# Инициализация лемматизатора
morph = pymorphy2.MorphAnalyzer()


# Функция для лемматизации текста
def lemmatize_text(word):
    lemmatized_word = morph.parse(word)
    return lemmatized_word[0].normal_form


# Загрузка данных
products_data = pd.read_csv('Data/products/male/categories.csv')
objects = products_data['Sub Category'].dropna().unique().tolist()

# Лемматизация объектов
filtered_objects = []
for item in objects:
    splited_items = re.split(r', ', item)
    for word in splited_items:
        if word not in ['и', 'для', 'на', 'в', '']:
            filtered_objects.append(lemmatize_text(word))

# Загрузка имен
male_names_data = pd.read_csv('Data/persons/male_names.csv', header=None)
male_names_data = male_names_data[1].tolist()

female_names_data = pd.read_csv("Data/persons/female_names.csv", header=None)
female_names_data = female_names_data[1].tolist()

# Списки имен
male_names = ["папа", "дедушка", "сын", "дядя", "брат", "парень", "братец", "братан", "мальчик", "отпрыск", "наследник",
              "отец", "батя"] + male_names_data
female_names = ["мама", "бабушка", "дочь", "сестра", "девушка", "мать", "мамочка", "девочка", "сестрица",
                "сестричка"] + female_names_data


# Загрузка событий
ru_events_data = pd.read_csv('Data/events/Russian_events.csv', header=None)
en_events_data = pd.read_csv('Data/events/International_events.csv', header=None)
events = ru_events_data[0].tolist() + en_events_data[0].tolist()

# Синонимы для аугментации
synonyms = {
    "подарок": ["презент", "сувенир", "дар"],
    "книга": ["том", "издание", "литературное произведение"],
    "игрушка": ["забавка", "детская игрушка", "игрушечный предмет"],
    "цветы": ["букеты", "флора", "растительность"],
    "одежда": ["наряд", "костюм", "гардероб"],
    "еда": ["пища", "блюда", "угощение"],
    "напиток": ["питье", "освежающий напиток", "коктейль"],
}


# Функция для замены слов синонимами
def replace_with_synonyms(text, synonyms):
    for word, syn_list in synonyms.items():
        if word in text:
            text = text.replace(word, random.choice(syn_list))
    return text


# Генерация размеченных данных
train_data = []
for _ in range(200000):
    example_type = random.choice(["original", "single", "no_entities"])

    if example_type == "original":
        # Выбор случайных сущностей
        name = random.choice(male_names + female_names)
        event = random.choice(events)
        obj = random.choice(filtered_objects)

        # Шаблоны предложений
        sentence_templates = [
            f"{name} отмечает {event} с {obj}",
            f"На {event} {name} пригласил всех родственников и подарил {obj}",
            f"{name} готовится к {event} и купил {obj}",
            f"{event} — любимый праздник {name}, и он всегда приносит {obj}",
            f"{name} и его семья празднуют {event} с {obj}?",
            f"В этом году {name} решил устроить {event} дома с {obj}?",
            f"{event} — это время, когда {name} собирает всех друзей и угощает {obj}?",
            f"В честь {event} {name} устроил вечеринку с {obj}?",
            f"{name} поздравляет всех с {event} и дарит {obj}?",
            f"{event} — это важный праздник для {name}, и он всегда покупает {obj}",
            f"В честь {event} {name} организовал праздничный ужин с {obj}",
            f"{name} собирается отметить {event} в кругу семьи и с {obj}",
            f"Все собрались на {event}, который устраивает {name}, и принесли {obj}",
            f"{event} — это особый день для {name} и его семьи, и они всегда готовят {obj}",
            f"{name} отметил {event} с размахом и купил {obj}?",
            f"Каждый год {name} празднует {event} с друзьями и угощает {obj}?",
            f"Я хочу купить {obj} {name} на {event}?",
            f"Я хочу купить для {name} на {event} {obj}?",
            f"Я хочу подарить {obj} для {name} на {event}",
            f"Я хочу подарить {name} на {event} {obj}",
            f"На {event} {name} не купил {obj}, потому что не успел",
            f"{name} подарил {obj} на {event}",
            f"{name} сказал, что на {event} он принесет {obj}",
            f"{obj} {name} на {event}",
            f"Купить {obj} {name} на {event}",
            f"{obj} на {event} для {name}",
        ]

        # Выбор случайного шаблона и замена синонимов
        sentence = random.choice(sentence_templates)
        sentence = replace_with_synonyms(sentence, synonyms)

        # Разметка сущностей
        entities = []

        # Разметка имени с учетом пола
        start_name = sentence.find(name)
        if start_name != -1:
            end_name = start_name + len(name)
            if name in male_names:
                entities.append((start_name, end_name, "PERSON_M"))  # Мужчина
            elif name in female_names:
                entities.append((start_name, end_name, "PERSON_F"))  # Женщина

        # Разметка события
        for word in event.split():
            start_event = sentence.find(word)
            if start_event != -1:
                end_event = start_event + len(word)
                if sentence[start_event:end_event] == word:  # Проверка корректности
                    entities.append((start_event, end_event, "EVENT"))

        # Разметка объекта
        for word in obj.split():
            start_obj = sentence.find(word)
            if start_obj != -1:
                end_obj = start_obj + len(word)
                if sentence[start_obj:end_obj] == word:  # Проверка корректности
                    entities.append((start_obj, end_obj, "OBJECT"))

        # Добавление примера
        train_data.append((sentence, {"entities": entities}))

    elif example_type == "single":
        # Примеры с одиночными сущностями
        single_type = random.choice(["name", "event", "object"])

        if single_type == "name":
            selected_names = random.sample(male_names + female_names, random.randint(1, 3))
            sentences = [', '.join(selected_names), "Для " + ", ".join(selected_names)]
            sentence = random.choice(sentences)
            entities = []
            for name in selected_names:
                # Разметка имени с учетом пола
                start_name = sentence.find(name)
                if start_name != -1:
                    end_name = start_name + len(name)
                    if name in male_names:
                        entities.append((start_name, end_name, "PERSON_M"))  # Мужчина
                    elif name in female_names:
                        entities.append((start_name, end_name, "PERSON_F"))  # Женщина

        elif single_type == "event":
            selected_events = random.sample(events, random.randint(1, 2))
            sentences = [", ".join(selected_events), "Сегодня отмечается " + ", ".join(selected_events)]
            sentence = random.choice(sentences)
            entities = []
            for event in selected_events:
                event_words = event.split()
                for word in event_words:
                    start_event = sentence.find(word)
                    if start_event != -1:
                        end_event = start_event + len(word)
                        entities.append((start_event, end_event, "EVENT"))

        elif single_type == "object":
            selected_objects = random.sample(filtered_objects, random.randint(1, 3))
            sentences = ["Купить " + ", ".join(selected_objects), ", ".join(selected_objects)]
            sentence = random.choice(sentences)
            entities = []
            for obj in selected_objects:
                object_words = obj.split()
                for word in object_words:
                    start_obj = sentence.find(word)
                    if start_obj != -1:
                        end_obj = start_obj + len(word)
                        entities.append((start_obj, end_obj, "OBJECT"))

        # Добавление примера
        train_data.append((sentence, {"entities": entities}))

    elif example_type == "no_entities":
        # Шаблоны предложений без сущностей
        sentence_templates_no_entities = [
            "Сегодня мы планируем провести важное мероприятие.",
            "В этом месяце ожидается много интересных событий.",
            "На выходных мы собираемся устроить встречу с друзьями.",
            "В следующем месяце будет много праздников и торжеств.",
            "Я хочу выбрать что-то особенное для этого случая.",
            "Нужно подумать, что подарить на праздник.",
            "В магазине много интересных вещей, которые стоит рассмотреть.",
            "Я собираюсь купить что-то для особого человека.",
            "Мы решили организовать небольшую вечеринку.",
            "В этом году мы хотим попробовать что-то новое.",
            "Я планирую провести время с близкими.",
            "На следующей неделе у нас запланирована поездка.",
            "Мне интересно, как лучше всего подготовиться к этому событию.",
            "Давайте обсудим, что мы можем сделать для улучшения ситуации.",
            "Я размышляю о том, как провести свободное время.",
            "Важно понимать, что каждый момент имеет значение.",
            "Я чувствую радость от предстоящих событий.",
            "В такие моменты всегда хочется быть рядом с близкими.",
            "Это время года всегда приносит особые эмоции.",
            "Я надеюсь, что все пройдет хорошо и принесет радость."
        ]
        # Примеры без сущностей
        sentence = random.choice(sentence_templates_no_entities)
        entities = []  # Пустой список сущностей

        # Добавление примера
        train_data.append((sentence, {"entities": entities}))

# Разделение на обучающую и валидационную выборки
random.shuffle(train_data)
split_index = int(len(train_data) * 0.8)
train_set = train_data[:split_index]
validation_set = train_data[split_index:]

# Запись обучающей выборки в файл
with open("Data/train_data.json", "w", encoding="utf-8") as f:
    json.dump(train_set, f, ensure_ascii=False, indent=4)

# Запись валидационной выборки в файл
with open("Data/validation_data.json", "w", encoding="utf-8") as f:
    json.dump(validation_set, f, ensure_ascii=False, indent=4)

print("Обучающая и валидационная выборки успешно созданы и записаны в файлы.")
