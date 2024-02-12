import transformers
from transformers import AutoModelForCausalLM, Pipeline

def inference_pipeline(pipeline_model: CreateInferencePipeline):
    model = Pipeline(pipeline_model.model, pipeline_model.tokenizer, device=pipeline_model.device)
    output = model(pipeline_model.input)
    return 

"""
TODO:
Functionalities:
+ Dataset import from huggingface, activeloop, custom_data, etc
+ tokenizer import
+ model import
+ inference [done]
+ option for using trl dataset builder [just an idea]
+ visualize the dataset
+ peft quantization [LoRA] [define only the LoRA config]
+ training args [include accelerate]
+ if we have multi-gpu support, we use accelerate [default] [optional]
"""