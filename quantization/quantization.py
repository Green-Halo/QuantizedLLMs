from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from huggingface_hub import create_repo, upload_folder

# Base model name
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_USERNAME = "chenyuezhang"

def load_and_quantize_model(quantization_type: str):
    """Load and quantize model."""
    print(f"Loading {quantization_type} quantized model...")

    # Bits and Bytes config
    if quantization_type == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        # 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto", 
            torch_dtype=torch.float16  
        )

    elif quantization_type == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        # 8-bit quantization 
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",  
            torch_dtype=torch.float16  
        )
    else:
        raise ValueError("Use '4bit' or '8bit'")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    return model, tokenizer

def push_to_hub(model, tokenizer, quantization_type: str):
    """Push model to Hugging Face."""
    repo_id = f"{HF_USERNAME}/llama3-8b-instruct-bnb-{quantization_type}"

    # Create repo
    create_repo(repo_id, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(f"./{repo_id}")
    tokenizer.save_pretrained(f"./{repo_id}")

    # Upload to Hugging Face
    upload_folder(
        repo_id=repo_id,
        folder_path=f"./{repo_id}",
        commit_message=f"Upload {quantization_type} quantized model"
    )
    print(f"Model {quantization_type} pushed to Hugging Face")


model_4bit, tokenizer_4bit = load_and_quantize_model("4bit")
push_to_hub(model_4bit, tokenizer_4bit, "4bit")


model_8bit, tokenizer_8bit = load_and_quantize_model("8bit")
push_to_hub(model_8bit, tokenizer_8bit, "8bit")
