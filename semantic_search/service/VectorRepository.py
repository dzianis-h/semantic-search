import hashlib

import psycopg2


class VectorRepository:
    def __init__(self):
        print("Setting-up DB")
        self.connection = psycopg2.connect(database="postgres", user='root', password='password', host="localhost",
                                           port=5432)
        self.embeddings_dimensions = 768
        self.reinit()

    def reinit(self):
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("""
CREATE TABLE IF NOT EXISTS embedding(
    id bigserial PRIMARY KEY,
    movie_id int not null,
    text_hash char(56) not null,
    embedding vector({}) not null
);
            """.format(self.embeddings_dimensions))
            self.connection.commit()

    def store_embedding(self, movie_id, text, embedding):
        with self.connection.cursor() as my_cursor:
            try:
                text_hash = hashlib.sha3_224(text.encode('UTF-8')).hexdigest()
                my_cursor.execute("INSERT INTO embedding (movie_id, text_hash, embedding) VALUES (%s, %s, %s)",
                                  (movie_id, text_hash, embedding))
                self.connection.commit()
            except psycopg2.Error as e:
                print(e)

    def semantic_search(self, embedding, max_results, max_distance):
        with self.connection.cursor() as cursor:
            cursor.execute("""
SELECT id, movie_id, embedding <=> '{}' as dist FROM embedding WHERE embedding <=> '{}' <= {} ORDER BY dist LIMIT {};
""".format(embedding, embedding, max_distance, max_results))
            return cursor.fetchall()
