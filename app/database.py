import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import DATABASE_URL, IS_PRODUCTION

logger = logging.getLogger("truthlens.database")

# Build engine with production pooling for PostgreSQL or thread-safety for SQLite
if DATABASE_URL.startswith("postgresql"):
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,    # Verifies connection viability before checkout
        pool_recycle=300,      # Recycles connections every 5 minutes to prevent stale cloud sockets
        echo=False
    )
    logger.info("Configured PostgreSQL connection pool with pre-ping validation.")
else:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False
    )
    logger.info("Configured SQLite local database connection.")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """FastAPI database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initializes database schema.
    Uses create_all to create missing tables safely without modifying or deleting existing records.
    """
    try:
        import app.models  # Ensure models are imported
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified and initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        return False

def check_db_connection() -> bool:
    """
    Health-check probe to verify live database connectivity without modifying state.
    Returns True if database is reachable and responsive, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning(f"Database health-check probe failed: {e}")
        return False
