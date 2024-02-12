from pydantic import BaseModel, Field, json
from typing import Optional, List, Union
from transformers import (
    PreTrainedModel, TFPreTrainedModel, PreTrainedTokenizer,
    SequenceFeatureExtractor, 
)
from transformers.image_processing_utils import BaseImageProcessor 
import torch

class CreateInferencePipeline(BaseModel):
    model: Union[PreTrainedModel, TFPreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer]
    feature_extractor: Optional[SequenceFeatureExtractor]
    image_processor: Optional[BaseImageProcessor]
    framework: Optional[str]
    task: Optional[str]
    num_workers: Optional[int] = 8
    batch_size: Optional[int] = 1
    device: Union[str, int] = -1
    torch_dtype: Union[str, torch.dtype]
    binary_output: Optional[bool] = False

