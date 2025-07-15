# 🎁 NLP-алгоритм подбора подарков | Gift Recommendation via NLP NER Model

## 📌 О проекте

Этот проект представляет собой алгоритм подбора подарков на основе NLP модели, решающей задачу **Named Entity Recognition (NER)**.  
Модель распознаёт в текстовых запросах три ключевые сущности:

- **Объект** — что или кто связан с подарком  
- **Событие** — повод или событие для подарка  
- **Получатель** — кому предназначен подарок  

На основе извлечённых сущностей алгоритм предлагает подходящие подарки.

## 📂 Структура репозитория

- **gen_data/** — скрипты для генерации и подготовки датасета

- **training_nlp/** — коды для обучения NER-модели

- **recommendation/** — реализация алгоритма извлечения сущностей и подбора подарков

- **try_requests/** — скрипты для проверки отправки данных на сервер и получения ответа модели

## 📌 About the Project

This project provides an algorithm for gift recommendation using an NLP model designed for Named Entity Recognition (NER).
The model identifies three key entities in the text input:

- **Object** — related person or item for the gift

- **Event** — occasion or event for the gift

- **Recipient** — the person who will receive the gift

Based on these extracted entities, the algorithm suggests suitable gift options.

## 📂 Repository Structure

- **gen_data/** — scripts for dataset generation and preparation

- **training_nlp/** — code for training the NER model

- **recommendation/** — implementation of the entity extraction algorithm and gift recommendation

- **try_requests/** — scripts to test server requests and responses
