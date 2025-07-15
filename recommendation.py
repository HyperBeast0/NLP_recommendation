from fastapi import FastAPI
from pydantic import BaseModel
import os
import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification
import torch
import pymorphy2
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from nltk import pos_tag, word_tokenize
from concurrent.futures import ThreadPoolExecutor

# Настройки
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForTokenClassification.from_pretrained("./my_rubert_model").to(device)
tokenizer = BertTokenizerFast.from_pretrained("DeepPavlov/rubert-base-cased")
morph = pymorphy2.MorphAnalyzer()
model_semantic = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

label_to_id = {"O": 0, "PERSON F": 1, "PERSON M": 2, "EVENT": 3, "OBJECT": 4}
id_to_label = {v: k for k, v in label_to_id.items()}

MIN_REVIEWS = 500
stop_words = {"[CLS]", "[SEP]", "я", "для", ",", "/", "-", "и", "или", "что", "нибудь"}


# Загрузка продуктов и кэширование эмбеддингов

def load_products(path_dir: str = "Data/products/product_v3/") -> pd.DataFrame:
    if os.path.exists("products_with_embeddings.pkl"):
        return pd.read_pickle("products_with_embeddings.pkl")

    dfs = []
    for filename in os.listdir(path_dir):
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(path_dir, filename), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        df = pd.DataFrame(data)
        if df.empty:
            continue
        df.columns = df.columns.str.strip()
        parts = os.path.splitext(filename)[0].split('_')
        df['category'] = parts[1].capitalize() if len(parts) >= 2 else "Unknown"
        df['gender'] = 'M' if parts[0].lower() == 'male' else 'F' if parts[0].lower() == 'female' else None
        df = df.rename(columns={'name': 'title', 'entity': 'description',
                                'feedbacks': 'review_count', 'reviewRating': 'rating'})
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['title', 'description', 'category', 'gender'])
    result['search_text'] = (result['title'].fillna('') + ' ' + result['description'].fillna('')).str.lower()
    result['embedding'] = result['search_text'].apply(lambda x: model_semantic.encode(x, convert_to_tensor=True))
    result['review_count'] = pd.to_numeric(result.get('review_count', 0), errors='coerce').fillna(0)
    result.to_pickle("products_with_embeddings.pkl")
    return result


products_df = load_products()


# NLP модуль

def lemmatize_text(text: str) -> str:
    return ' '.join(morph.parse(w)[0].normal_form for w in text.split())


def predict_entities(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    labels = logits.argmax(dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    words, labs, curr, lab_curr = [], [], '', None
    for tok, lab in zip(tokens, labels):
        word = tok[2:] if tok.startswith('##') else tok
        if not word.strip():
            continue
        if tok.startswith('##'):
            curr += word
        else:
            if curr:
                words.append(curr)
                labs.append(lab_curr)
            curr, lab_curr = word, lab
    if curr:
        words.append(curr)
        labs.append(lab_curr)
    return words, labs


def expand_keywords_semantically(objects, df, top_n=30):
    if not objects:
        return []
    query_emb = model_semantic.encode(' '.join(objects), convert_to_tensor=True)
    df = df.copy()
    df['similarity'] = df['embedding'].apply(lambda x: util.pytorch_cos_sim(query_emb, x).item())
    similar = df.sort_values('similarity', ascending=False).head(top_n)
    phrases = []
    for title in similar['title'].fillna(''):
        tagged = pos_tag(word_tokenize(title))
        phrase = []
        for word, tag in tagged:
            if tag.startswith('JJ') or tag.startswith('NN'):
                phrase.append(word.lower())
            else:
                if phrase:
                    phrases.append(' '.join(phrase))
                    phrase = []
        if phrase:
            phrases.append(' '.join(phrase))
    return list(dict.fromkeys([p for p in phrases if 2 <= len(p.split()) <= 4]))


def recommend_categories(df, keys, top_n=3):
    tokenized = [text.split() for text in df['search_text']]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(keys)
    df = df.copy()
    df['score'] = scores
    df = df[df['review_count'] >= MIN_REVIEWS]
    agg = df.groupby('category')['score'].sum().sort_values(ascending=False)
    top_cats = list(agg.head(top_n).index)
    if len(top_cats) < top_n:
        fallback = df['category'].value_counts().index.difference(top_cats).tolist()
        top_cats += fallback[:top_n - len(top_cats)]
    return top_cats


def recommend_products(df, keys):
    tokenized = [text.split() for text in df['search_text']]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(keys)
    df = df.copy()
    df['score'] = scores
    df = df[(df['score'] > 0) & (df['review_count'] >= MIN_REVIEWS)]
    return df.sort_values('score', ascending=False)


# FastAPI

app = FastAPI()


class Query(BaseModel):
    text: str


class Entity(BaseModel):
    word: str
    label: str


class Product(BaseModel):
    id: str
    title: str


class CategoryProducts(BaseModel):
    category: str
    products: list[Product]


class Recommendation(BaseModel):
    categories: list[str]
    products: list[CategoryProducts]


class Response(BaseModel):
    entities: list[Entity]
    keywords: list[str]
    recommendations: Recommendation


@app.post('/recommend', response_model=Response)
async def recommend(query: Query):
    text = query.text
    lem_text = lemmatize_text(text)
    words, labs = predict_entities(lem_text)
    entities = [Entity(word=w, label=id_to_label.get(l, 'O')) for w, l in zip(words, labs)]

    ev_phrases, obj_phrases = [], []
    i = 0
    while i < len(words):
        if labs[i] in (label_to_id['EVENT'], label_to_id['OBJECT']):
            tag = labs[i]
            phrase = words[i]
            j = i + 1
            while j < len(words) and labs[j] == tag:
                phrase += ' ' + words[j]
                j += 1
            if tag == label_to_id['EVENT']:
                ev_phrases.append(phrase)
            else:
                obj_phrases.append(phrase)
            i = j
        else:
            i += 1

    objects_list = [' '.join([t for t in obj.split() if t not in stop_words]) for obj in obj_phrases if obj]
    events_list = [' '.join([t for t in ev.split() if t not in stop_words]) for ev in ev_phrases if ev]
    person_list = [words[i] for i, l in enumerate(labs) if l in (label_to_id['PERSON M'], label_to_id['PERSON F'])]

    keywords = list(dict.fromkeys(
        [k for k in objects_list + events_list + person_list if k] +
        [w for w, l in zip(words, labs) if l in (label_to_id['OBJECT'], label_to_id['EVENT'], label_to_id['PERSON M'],
                                                 label_to_id['PERSON F']) and w not in stop_words]
    ))

    keywords_weighted = []
    for k in keywords:
        if k in objects_list:
            keywords_weighted.extend([k] * 5)
        elif k in events_list:
            keywords_weighted.extend([k] * 3)
        elif k in person_list:
            keywords_weighted.extend([k] * 4)
        else:
            keywords_weighted.append(k)

    df_g = products_df
    if label_to_id['PERSON F'] in labs:
        df_g = products_df[products_df['gender'] == 'F']
    elif label_to_id['PERSON M'] in labs:
        df_g = products_df[products_df['gender'] == 'M']

    semantic_expansion = expand_keywords_semantically(objects_list, df_g)
    keywords_weighted.extend(semantic_expansion * 2)

    if not keywords_weighted:
        cats = df_g['category'].value_counts().head(3).index.tolist()
        return Response(entities=entities, keywords=[], recommendations=Recommendation(categories=cats, products=[]))

    cats = recommend_categories(df_g, keywords_weighted)

    def fetch_products(cat):
        prods_df = recommend_products(df_g[df_g['category'] == cat], keywords_weighted)
        return CategoryProducts(
            category=cat,
            products=[Product(id=str(r['id']), title=r['title']) for _, r in prods_df.iterrows()]
        )

    with ThreadPoolExecutor() as executor:
        products_by_category = list(executor.map(fetch_products, cats))

    return Response(
        entities=entities,
        keywords=keywords + semantic_expansion,
        recommendations=Recommendation(categories=cats, products=products_by_category)
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
