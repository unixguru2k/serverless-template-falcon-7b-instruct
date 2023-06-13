# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True)


if __name__ == "__main__":
    download_model()
