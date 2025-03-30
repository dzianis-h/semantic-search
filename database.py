import psycopg2

connection = psycopg2.connect(database="postgres", user='root', password='password', host="localhost", port=5432)


def init_db():
    print("Setting-up DB")
    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS embedding")
        cursor.execute(
            "CREATE TABLE embedding(id bigserial PRIMARY KEY, text varchar(2048) not null, embedding vector(768) not null)")
        connection.commit()


def store_embedding(text, embedding):
    with connection.cursor() as my_cursor:
        my_cursor.execute("INSERT INTO embedding (text, embedding) VALUES (%s, %s)", (text, embedding))
        connection.commit()
