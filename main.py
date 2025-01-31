import time
import re
import os
import requests
from typing import List, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, HttpUrl
from utils.logger import setup_logger
from contextlib import asynccontextmanager


from dotenv import load_dotenv


load_dotenv()
IAM_TOKEN = os.getenv("IAM_TOKEN")
FOLDER_ID = os.getenv("FOLDER_ID")

YANDEX_GPT_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


logger = setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """ Управление жизненным циклом приложения """
    yield


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    id: int
    query: str


class PredictionResponse(BaseModel):
    id: int
    answer: str | None
    reasoning: str
    sources: List[HttpUrl]


@app.middleware("http")
async def log_requests(request: Request, call_next):

    start_time = time.time()
    body = await request.body()

    if logger:
        logger.info(
            f"Incoming request: {request.method} {request.url}\n"
            f"Request body: {body.decode()}"
        )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    if logger:
        logger.info(
            f"Request completed: {request.method} {request.url}\n"
            f"Status: {response.status_code}\n"
            f"Response body: {response_body.decode()}\n"
            f"Duration: {process_time:.3f}s"
        )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )



def get_ai_response(query: str) -> str:
    headers = {
        "Authorization": f"Bearer {IAM_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt-lite",
        "completionOptions": {"stream": False, "temperature": 0.7, "maxTokens": "2000"},
        "messages": [{"role": "user", "text": query}]
    }
    response = requests.post(YANDEX_GPT_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["result"]["alternatives"][0]["message"]["text"]
    return "Не удалось получить ответ от модели."


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):

    try:
        if logger:
            logger.info(f"Processing prediction request with id: {body.id}")

        answer = get_ai_response(body.query)

        response = PredictionResponse(
            id=body.id,
            answer=answer,
            reasoning="Ответ получен с помощью YandexGPT.",
            sources=[]
        )

        if logger:
            logger.info(f"Successfully processed request {body.id}")

        return response

    except Exception as e:
        if logger:
            logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    PORT = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
