DB_USER = 'postgres'
DB_PASSWORD = 'password'
DB_DATABASE = 'postgres'
DB_HOST = 'localhost'
CONN_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_DATABASE}?sslmode=disable"
CONN_STRING_PSYCOPG3 = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_DATABASE}?sslmode=disable"



TEMP_DIR = 'temp'