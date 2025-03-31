import psycopg2

connection = psycopg2.connect(database="postgres", user='root', password='password', host="localhost", port=5432)

embeddings_dimensions = 256


def init_db():
    print("Setting-up DB")
    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS embedding")
        cursor.execute(
            "CREATE TABLE embedding(id bigserial PRIMARY KEY, movie_id int not null, text varchar(2048) not null, embedding vector({}) not null)".format(
                embeddings_dimensions))
        connection.commit()


def store_embedding(movie_id, text, embedding):
    with connection.cursor() as my_cursor:
        try:
            my_cursor.execute("INSERT INTO embedding (movie_id, text, embedding) VALUES (%s, %s, %s)",
                              (movie_id, text, embedding))
            connection.commit()
        except psycopg2.Error as e:
            print(e)


def semantic_search(embedding):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, movie_id, text, (embedding <=> '{}') as dist FROM embedding ORDER BY dist LIMIT 6;".format(
                embedding))
        return cursor.fetchall()
