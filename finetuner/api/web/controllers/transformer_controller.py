from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from ..db import get_session
from ..dto.request_model import CreateInferencePipeline
from ..services import create_inference_pipeline_model

router = APIRouter()

@router.get("/models")
async def load_model(session: AsyncSession = Depends(get_session)):
    pass

@router.get("/models/{model_id}")
async def load_tokenizer(model_id: str, session: AsyncSession = Depends(get_session)):
    pass

@router.post("/models")
async def save_model(session: AsyncSession = Depends(get_session)):
    pass

@router.post("/models/inference")
async def inference_model(pipeline_model: CreateInferencePipeline, session: AsyncSession = Depends(get_session)):
    try:
        return await create_inference_pipeline_model(pipeline_model, session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    