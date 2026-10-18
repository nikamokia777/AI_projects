from sqlalchemy import create_engine

DATABASE_URL = "postgresql://neondb_owner:npg_fzc8Q9UleRxW@ep-wispy-grass-bafudft8-pooler.c-8.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

engine = create_engine(DATABASE_URL)