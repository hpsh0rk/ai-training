import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()


def get_embedding(text, model="text-embedding-ada-002"):  # model = "deployment_name"
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def search_docs(df, user_query, top_n=4, to_print=True):
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    embedding = get_embedding(
        user_query,
        model="text-embedding-ada-002",  # model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = df.sort_values("similarities", ascending=False).head(top_n)
    if to_print:
        print(res)
    return res


def embedding_and_search():
    # load data
    df = pd.read_csv("dataset/Kaggle related questions on Qoura - Questions.csv")
    print(df.shape)
    print(df.head())
    print("==================================================")

    # embedding knowledge
    vec_base = []
    # for v in df.head().itertuples(index=True,name='DataRow'):
    for v in df.head().itertuples():
        emb = get_embedding(v.Questions)
        im = {
            "question": v.Questions,
            "embedding": emb,
            "answer": v.Link,
        }
        vec_base.append(im)

    # embedding question
    query = "is kaggle alive?"
    q_emb = get_embedding(query)

    print(vec_base[1]["question"], vec_base[1]["answer"])

    # search
    arr = np.array([v["embedding"] for v in vec_base])
    print(arr.shape)

    q_arr = np.expand_dims(q_emb, 0)
    print(q_arr.shape)
    print(cosine_similarity(arr, q_arr))
    """
        (1166, 4)
                                                Questions  ...                                               Link
        0  How do I start participating in Kaggle competi...  ...  /How-do-I-start-participating-in-Kaggle-compet...
        1                                    Is Kaggle dead?  ...                                    /Is-Kaggle-dead
        2       How should a beginner get started on Kaggle?  ...       /How-should-a-beginner-get-started-on-Kaggle
        3              What are some alternatives to Kaggle?  ...              /What-are-some-alternatives-to-Kaggle
        4  What Kaggle competitions should a beginner sta...  ...  /What-Kaggle-competitions-should-a-beginner-st...

        [5 rows x 4 columns]
        ==================================================
        [0.7880016151454352, 0.9542302479388193, 0.822751797046233, 0.8398752285928417, 0.8109667304826277]
        Is Kaggle dead? /Is-Kaggle-dead
        (5, 1536)
        (1, 1536)
        [[0.78800162]
        [0.95423025]
        [0.8227518 ]
        [0.83987523]
        [0.81096673]]
    """
