import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer

    # Flan-T5 version, if changed be sure to update in download.py too
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
    model.half().to(torch.cuda.current_device())

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer
    system_prompt = "" 
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    max_new_tokens= model_inputs.get('max_new_tokens', 64)
    temperature= model_inputs.get('temperature', 0.7)
    full_prompt = system_prompt + prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokens = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    do_sample=True,
    stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )
    result = {"output":tokenizer.decode(tokens[0], skip_special_tokens=True)}

    # Return the results as a dictionary
    return result
