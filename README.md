# LLaMA3-8B-Lawyer

This project involves fine-tuning the LLaMA3-8B model using the `dzunggg/legal-qa-v1` dataset to create a high-performing legal question-answering AI. The fine-tuning was conducted with the LLaMA-Factory toolkit on a single NVIDIA L20-48G GPU. The fine-tuned model has been uploaded to Hugging Face and is available at [StevenChen16/llama3-8b-Lawyer](https://huggingface.co/StevenChen16/llama3-8b-Lawyer).

## Project Overview

The primary goal of this project was to create a high-performing legal question-answering model based on LLaMA3-8B. The AI model can function like a lawyer, asking detailed questions about the case background and making judgments based on the provided information.

## Repository Contents

- `gr2.py`: A script to call the model using Gradio.
- `app.py`: A script to call the model using the Flask framework.
- `app_gr.py`: A script to construct the frontend by calling the link generated by `app.py`.
- `Finetune_Llama3_70b_with_LLaMA_Factory.ipynb`: Jupyter notebook for fine-tuning the model, similar to the Google Colab notebook.

## Fine-Tuning Details

### Model
- Base Model: `meta-llama/Meta-Llama-3-8B`
- Fine-Tuned Model: `StevenChen16/llama3-8b-Lawyer`

### Dataset
- Dataset Used: `dzunggg/legal-qa-v1`

### Training Configuration

```python
args = dict(
  stage="sft",                        # do supervised fine-tuning
  do_train=True,
  model_name_or_path="meta-llama/Meta-Llama-3-8B", # use bnb-4bit-quantized Llama-3-8B-Instruct model
  dataset="legal_qa_v1_train",             # use legal_qa_v1_train dataset
  template="llama3",                     # use llama3 prompt template
  finetuning_type="lora",                   # use LoRA adapters to save memory
  lora_target="all",                     # attach LoRA adapters to all linear layers
  output_dir="llama3_lora",                  # the path to save LoRA adapters
  per_device_train_batch_size=8,               # the batch size
  gradient_accumulation_steps=6,               # the gradient accumulation steps
  lr_scheduler_type="cosine",                 # use cosine learning rate scheduler
  logging_steps=10,                      # log every 10 steps
  warmup_ratio=0.1,                      # use warmup scheduler
  save_steps=1000,                      # save checkpoint every 1000 steps
  learning_rate=1e-4,                     # the learning rate
  num_train_epochs=10.0,                    # the epochs of training
  max_samples=500,                      # use 500 examples in each dataset
  max_grad_norm=1.0,                     # clip gradient norm to 1.0
  quantization_bit=8,                     # use 8-bit quantization
  loraplus_lr_ratio=16.0,                   # use LoRA+ algorithm with lambda=16.0
  use_unsloth=True,                      # use UnslothAI's LoRA optimization for 2x faster training
  fp16=True,                         # use float16 mixed precision training
  overwrite_output_dir=True,
)
```

### Hardware
- GPU: NVIDIA L20-48G

## Usage

### Gradio Interface

You can use the Gradio interface to interact with the model by running `gr2.py`:

```bash
python gr2.py
```

### Flask Application

To use the Flask application to call the model, run `app.py`:

```bash
python app.py
```

You can then construct the frontend by running `app_gr.py`, which calls the link generated by `app.py`:

```bash
python app_gr.py
```

### Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "StevenChen16/llama3-8b-Lawyer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example usage
input_text = "Your legal question here."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

### Example Interaction

The model can engage in a detailed interaction, simulating the behavior of a lawyer. Provide the case background, and the model will ask for more details to make informed judgments.

```python
input_text = "I have a contract dispute where the other party did not deliver the promised goods."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

Output:
```
Can you provide more details about the contract terms and the goods that were supposed to be delivered? Were there any specific deadlines mentioned in the contract?
```

## Training Notebook and Repository

- Training Notebook: [Google Colab Notebook](https://colab.research.google.com/drive/14oOHgdML0dCL8Ku7PimU-u1KGoQbAjNP?usp=sharing)
- GitHub Repository: [lawyer-llama3-8b](https://github.com/StevenChen16/lawyer-llama3-8b.git)
- Model: [StevenChen16/llama3-8b-Lawyer](https://huggingface.co/StevenChen16/llama3-8b-Lawyer)
- Huggingface Space: [StevenChen16/llama3-8b-Lawyer](https://huggingface.co/spaces/StevenChen16/llama3-8b-Lawyer)

## Results

The fine-tuned model has shown promising results in understanding and answering legal questions. By leveraging advanced techniques such as LoRA and UnslothAI optimizations, the training process was efficient and effective, ensuring a high-quality model output.

## Acknowledgements

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Dataset: `dzunggg/legal-qa-v1`
- Base Model: `meta-llama/Meta-Llama-3-8B`
- Hosted on [Hugging Face](https://huggingface.co/StevenChen16/llama3-8b-Lawyer)

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for detail
