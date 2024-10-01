import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
import gc
from huggingface_hub import HfApi, HfFolder
from peft import prepare_model_for_kbit_training
import transformers
from peft import PeftModel
import ray
import ray.train.huggingface.transformers
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, get_device
#from trl import SFTTrainer

def training_func():
    hf_user = 'yajuvendra'
    hf_token = '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'

    HfFolder.save_token(hf_token)

    # The model that you want to train from the Hugging Face hub
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    #model_id = "meta-llama/Llama-2-13b-chat-hf"

    # The instruction dataset to use
    #dataset_name = "mlabonne/guanaco-llama2-1k"

    # Fine-tuned model name
    #new_model = "Llama-2-7b-chat-finetune"

    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = 64

    # Alpha parameter for LoRA scaling
    lora_alpha = 16

    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Output directory where the model predictions and checkpoints will be stored
    output_dir = "./results"

    # Number of training epochs
    num_train_epochs = 1

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = False

    # Batch size per GPU for training
    per_device_train_batch_size = 4

    # Batch size per GPU for evaluation
    per_device_eval_batch_size = 4

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate = 2e-4

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule
    lr_scheduler_type = "cosine"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # Save checkpoint every X updates steps
    save_steps = 0

    # Log every X updates steps
    logging_steps = 25

    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length = None

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False

    # Load the entire model on the GPU 0
    device_map = {"": torch.cuda.current_device()}

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


    # Load dataset (you can process it here)
    #data = load_dataset(dataset_name, split="train")
    data = load_dataset("Abirate/english_quotes")
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    # Load LoRA configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        # target_modules=["query_key_value"],
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ],  # specific to Llama models.
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            ddp_find_unused_parameters=False,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )


    # Train model
    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)

    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

    model.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        )

    trainer.train()

    base_model_name = model_id.split("/")[-1]

    # Define the save and push paths
    adapter_model = f"{hf_user}/{base_model_name}-fine-tuned-adapters"
    new_model = f"{hf_user}/{base_model_name}-fine-tuned"

    # Save the model
    model.save_pretrained(adapter_model, push_to_hub=True, use_auth_token=True)

    # Push the model to the hub
    model.push_to_hub(adapter_model, use_auth_token=True)

    # Save trained model
    trainer.model.save_pretrained(new_model)

    # Save the model
    model.save_pretrained(adapter_model, push_to_hub=True, use_auth_token=True)

    # Push the model to the hub
    model.push_to_hub(adapter_model, use_auth_token=True)

    # load perf model with new adapters
    model = PeftModel.from_pretrained(
        model,
        adapter_model,
    )

    model = model.merge_and_unload()  # merge adapters with the base model.
    model.push_to_hub(new_model, use_auth_token=True, max_shard_size="5GB")
    cache_dir = "cache_dir"

    # Empty VRAM
    del model
    del trainer
    gc.collect()
    gc.collect()

    # reload the base model (you might need a pro subscription for this because you may need a high RAM environment for the 13B model since this is loading the full original model, not quantized)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    )

    # load perf model with new adapters
    model = PeftModel.from_pretrained(
        model,
        adapter_model,
    )

    model = model.merge_and_unload()  # merge adapters with the base model.

    model.push_to_hub(new_model, use_auth_token=True, max_shard_size="5GB")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.push_to_hub(new_model, use_auth_token=True)

ray_trainer = TorchTrainer(
    training_func,
    scaling_config=ScalingConfig(
        num_workers=3,  # Set this to the number of worker nodes
        use_gpu=True,
    ),
    # [4a] If running in a multi-node cluster, this is where you
    # should configure the run's persistent storage that is accessible
    # across all worker nodes.
)
result: ray.train.Result = ray_trainer.fit()