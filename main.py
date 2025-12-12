from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from agent import run_agent
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/solve")
async def solve(request: Request):
    data = await request.json()

    os.environ["url"] = data["url"]
    os.environ["EMAIL"] = data["email"]
    os.environ["SECRET"] = data["secret"]

    run_agent(data["url"])

    return {"status": "submitted"}
