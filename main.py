from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.qdrant import ensure_collection
    from app.models import Base
    from app.core.database import engine
    try:
        Base.metadata.create_all(bind=engine)
        ensure_collection()
    except Exception as e:
        print(f"Qdrant init warning: {e}")
    yield


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(api_router)


@app.get("/")
def root():
    return {"service": settings.app_name, "docs": "/docs"}
