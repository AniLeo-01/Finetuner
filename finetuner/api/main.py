from fastapi import FastAPI
from contextlib import asynccontextmanager
# import the controllers
# import the db models
from api.web.db import disconnect_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    await init_db()
    yield
    print("Shutting down...")
    await disconnect_db()

app = FastAPI(title = 'Finetuner', version="1.0.0", lifespan=lifespan)

app.include_router()