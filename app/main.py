from dotenv import load_dotenv
load_dotenv()

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db import mongo
from app.routers import chat, threads


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo.connect()
    yield
    mongo.disconnect()


app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:3000"]
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return "Hello World"
app.include_router(threads.router)
app.include_router(chat.router)

#emoty
