import torch
import transformers
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
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct")
    model.half().to(torch.cuda.current_device())


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model
    global tokenizer

    output = ""
    # system_prompt = ""
    prompt = model_inputs.get('prompt', None)
    if prompt is None:
        return {'message': "No prompt provided"}
    # max_new_tokens= model_inputs.get('max_new_tokens', 64)
    # temperature= model_inputs.get('temperature', 0.7)
    # full_prompt = system_prompt + prompt

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    sequences = pipeline(
        prompt,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
        output += seq['generated_text']

    result = {"output": output}

    # Return the results as a dictionary
    return result
