# qa bot
## what can learn from this
### emb_and_search.py
1. how to reqeust openai
2. how to use embedding and match similar text
### emb_and_search_redis.py
1. how to use redis storage vectors and query


## how to run
### create new env & import package
```
python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt
```
### set openai key
```
export OPENAI_API_KEY="sk-xxx"
```
### install redis
```
docker run -d -p 6379:6379 -it redis/redis-stack:latest
```
