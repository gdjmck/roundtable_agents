from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
target = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=target,
    allow_credentials=True,
    allow_methods=target,
    allow_headers=target,
)

__all__ = [
    "app"
]
