import psycopg2

connection = psycopg2.connect(database="postgres", user='root', password='password', host="localhost", port=5432)


def init_db():
    print("Setting-up DB")
    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS embedding")
        cursor.execute(
            "CREATE TABLE embedding(id bigserial PRIMARY KEY, kkid int not null, text varchar(2048) not null, embedding vector(768) not null)")
        connection.commit()


def store_embedding(kkid, text, embedding):
    with connection.cursor() as my_cursor:
        my_cursor.execute("INSERT INTO embedding (kkid, text, embedding) VALUES (%s, %s, %s)", (kkid, text, embedding))
        connection.commit()

def semantic_search(embedding):
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, kkid, text, (embedding <=> '{}') as dist FROM embedding ORDER BY dist LIMIT 6;".format(embedding))
        return cursor.fetchall()