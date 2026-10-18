from sqlalchemy import Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

from database import engine

class Base(DeclarativeBase):
    pass

class DocumentChunk(Base):
    __tablename__ = "hotel_info_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(3072))
    meta: Mapped[dict] = mapped_column('metadata',JSONB)

if __name__ == "__main__":
    Base.metadata.create_all(engine)