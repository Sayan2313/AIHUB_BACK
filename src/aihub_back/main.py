import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request , status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware

from aihub_back.core.exceptions import AppException
from aihub_back.logs import get_logger

from aihub_back.routes.llm import llm_router

# ----------------------------------Application----------------------------------

app = FastAPI()
logger = get_logger()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/api/models/info")
async def models_info():
    file_path = Path("src/aihub_back/models/info.json")
    if file_path.exists():
        return json.loads(file_path.read_text())
    else:
        return HTTPException(status_code=404, detail="File not found")


# ----------------------------------Routes----------------------------------
app.include_router(llm_router)


#----------------------------------Exceptions----------------------------------

@app.exception_handler(AppException)
async def handled_exception_handler(request: Request, exc: AppException):
    logger.exception(f"{exc.message} | Path: {request.url.path}")
    return JSONResponse(status_code=exc.status_code, content={"detail": "internal server error"})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Path: {request.url.path} | Unhandled Error -> {exc!s}")
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "internal server error"})

