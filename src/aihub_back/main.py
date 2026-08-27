import json
from json import JSONDecodeError
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from aihub_back.core.exceptions import AppException
from aihub_back.logs import get_logger
from aihub_back.routes.llm import llm_router
from aihub_back.schemas.schemas import ModelsInfoSchema

# ----------------------------------Application----------------------------------

app = FastAPI()
logger = get_logger()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.1.4:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"]
)


@app.get("/api/models/info")
async def models_info():
    file_path = Path("src/aihub_back/models/info.json")
    if file_path.exists():
        try:
            data = json.loads(file_path.read_text())
            return ModelsInfoSchema.model_validate(data)
        except JSONDecodeError:
            raise AppException("Wrong Formatted Json File", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except ValidationError:
            raise AppException("Data Validation Error",status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
    else:
        return HTTPException(status_code=404, detail="File not found")


# ----------------------------------Routes----------------------------------
app.include_router(llm_router)


#----------------------------------Exceptions----------------------------------

@app.exception_handler(AppException)
async def handled_exception_handler(request: Request, exc: AppException):
    logger.exception(f"Path: {request.url.path} | {exc!s}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Path: {request.url.path} | Unhandled Error -> {exc!s}")
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "internal server error"})

