from redis.commands.search.query import Query
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition
import redis
import pandas as pd
from openai import OpenAI
import numpy as np


client = OpenAI()


def get_embedding(text, model="text-embedding-ada-002"):  # model = "deployment_name"
    return client.embeddings.create(input=[text], model=model).data[0].embedding


r = redis.Redis()
INDEX_NAME = "faq"
index = r.ft(INDEX_NAME)
# text-embedding-ada-002 embedding dim is 1536
VECTOR_DIM = 1536


def embedding_and_search_with_redis_drop_index():
    index.dropindex(delete_documents=True)


def embedding_and_search_with_redis_save_data():
    question = TextField(name="question")
    answer = TextField(name="answer")
    embedding = VectorField(
        name="embedding",
        algorithm="HNSW",
        attributes={"TYPE": "FLOAT32", "DIM": VECTOR_DIM, "DISTANCE_METRIC": "COSINE"},
    )
    schema = (question, embedding, answer)
    index = r.ft(INDEX_NAME)
    try:
        info = index.info()
    except:
        index.create_index(
            schema, definition=IndexDefinition(prefix=[INDEX_NAME + "-"])
        )
        # index.dropindex(delete_documents=True)

    # load data
    df = pd.read_csv("dataset/Kaggle related questions on Qoura - Questions.csv")
    print(df.head())
    # save data to redis
    for v in df.head().itertuples():
        emb = get_embedding(v.Questions)
        # Note that redis needs to store bytes or strings
        emb = np.array(emb, dtype=np.float32).tobytes()
        im = {"question": v.Questions, "embedding": emb, "answer": v.Link}
        r.hset(name=f"{INDEX_NAME}-{v.Index}", mapping=im)
        print(im)


def embedding_and_search_with_redis_query():
    # get question embedding
    query = "kaggle alive?"
    embed_query = get_embedding(query)
    params_dict = {
        "query_embedding": np.array(embed_query).astype(dtype=np.float32).tobytes()
    }
    print(params_dict)
    # query in redis
    k = 3
    # {some filter query}=>[ KNN {num|$num} @vector_field $query_vec]
    base_query = f"* => [KNN {k} @embedding $query_embedding AS score]"
    return_fields = ["question", "answer", "score"]
    query = (
        Query(base_query)
        .return_fields(*return_fields)
        .sort_by("score")
        .paging(0, k)
        .dialect(2)
    )
    print(f"query:{query}")
    res = index.search(query, params_dict)
    print(f"res:{res}")
    for i, doc in enumerate(res.docs):
        similarity = 1 - float(doc.score)
        print(
            f"{doc.id}, {doc.question}, {doc.answer} (Similarity: {round(similarity ,3) }) {doc.score}"
        )
    """
    res:Result{3 total, docs: [Document {'id': 'faq-1', 'payload': None, 'score': '0.0710220336914', 'question': 'Is Kaggle dead?', 'answer': '/Is-Kaggle-dead'}, Document {'id': 'faq-3', 'payload': None, 'score': '0.156078279018', 'question': 'What are some alternatives to Kaggle?', 'answer': '/What-are-some-alternatives-to-Kaggle'}, Document {'id': 'faq-2', 'payload': None, 'score': '0.166731417179', 'question': 'How should a beginner get started on Kaggle?', 'answer': '/How-should-a-beginner-get-started-on-Kaggle'}]}

    faq-1, Is Kaggle dead?, /Is-Kaggle-dead (Similarity: 0.929) 0.0710220336914
    faq-3, What are some alternatives to Kaggle?, /What-are-some-alternatives-to-Kaggle (Similarity: 0.844) 0.156078279018
    faq-2, How should a beginner get started on Kaggle?, /How-should-a-beginner-get-started-on-Kaggle (Similarity: 0.833) 0.166731417179
    """
