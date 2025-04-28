import hashlib

import psycopg2
from psycopg2._psycopg import cursor


class VectorRepository:
    def __init__(self):
        print("Setting-up DB")
        self.connection = psycopg2.connect(database="postgres", user='root', password='password', host="localhost",
                                           port=5432)
        self.embeddings_dimensions = 1024
        self.reinit()

    def reinit(self):
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("""
CREATE TABLE IF NOT EXISTS embedding(
    id bigserial PRIMARY KEY,
    movie_id int not null,
    chunk varchar(31) not null,
    text_hash char(56) not null,
    embedding vector({}) not null
);
            """.format(self.embeddings_dimensions))
            self.connection.commit()

    def truncate_embeddings(self):
        with self.connection.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE embedding;")
            self.connection.commit()

    def store_embeddings(self, movie_id, chunks, texts, embeddings):
        if len(chunks) == 0:
            return

        query = "INSERT INTO embedding (movie_id, chunk, text_hash, embedding) VALUES"
        args = []
        for idx in range(len(chunks)):
            text_hash = hashlib.sha3_224(texts[idx].encode('UTF-8')).hexdigest()
            query = query + "(%s, %s, %s, %s),"
            args.extend((movie_id, chunks[idx], text_hash, embeddings[idx]))

        query = query[:-1] + ';'

        with self.connection.cursor() as cursor:
            try:
                cursor.execute(query, args)
                self.connection.commit()
            except psycopg2.Error as e:
                print(e)

    def semantic_search(self, embedding, max_results, max_distance):
        with self.connection.cursor() as cursor:
            cursor.execute("""
SELECT id, movie_id, chunk, embedding <=> '{}' as dist FROM embedding 
WHERE embedding <=> '{}' <= {} 
ORDER BY dist LIMIT {};
""".format(embedding, embedding, max_distance, max_results))
            return cursor.fetchall()
