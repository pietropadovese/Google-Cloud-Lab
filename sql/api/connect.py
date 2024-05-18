import os
from sqlalchemy import create_engine

# Credentials
credentials = [
    'DB_USER',
    'DB_PASS',
    'DB_NAME',
    'DB_HOST'
]

# Check environment
db = [os.getenv(c) for c in credentials]
if not any(db):
    raise ValueError(
        "Set the environment variables DB_USER, DB_PASS, DB_NAME, DB_HOST "
        "to run this API with a PostgreSQL database"
    )

# Create engine
engine = create_engine(
    f"postgresql+psycopg2://{db[0]}:{db[1]}@/{db[2]}?host={db[3]}"
)

# Test connection
with engine.connect() as con:
    pass
