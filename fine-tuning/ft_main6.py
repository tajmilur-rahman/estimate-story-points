import re
import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, median_absolute_error
from transformers import pipeline

from dataset_split import *
from retriever_def import *
from generate_def import *

# ==== Config ====
model_id = "meta-llama/Llama-3.2-3B-Instruct"
output_dir = "./lora-llama3-finetuned"
train_file = "Another fine-tuning/train.jsonl"
test_file = "Another fine-tuning/test.jsonl"
use_flash_attn = False  # If supported by hardware


# Load dataset
summarized_file = "/u1/users/fnv012/SPLLAMA3/Another fine-tuning/FABsummarized.csv"
train_df, test_df = format_data(summarized_file)


# ==== Load and preprocess dataset ====
def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]
    

train_data = load_jsonl(train_file)
test_data = load_jsonl(test_file)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# ==== Tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# ==== Load model with 4-bit quantization ====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if use_flash_attn else None,
)

# ==== LoRA config ====
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)

# for param in model.parameters():
#     if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
#         param.requires_grad = True

# ==== Training Arguments ====
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     gradient_accumulation_steps=4,
#     save_strategy="epoch",
#     logging_dir=f"{output_dir}/logs",
#     num_train_epochs=3,
#     learning_rate=1e-4,
#     bf16=True,
#     optim="paged_adamw_8bit",
#     logging_steps=10,
#     save_total_limit=2,
#     push_to_hub=False,
#     report_to="none"
# )

# ==== Training Arguments ====
training_args = SFTConfig(
    num_train_epochs=4,
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    lr_scheduler_type="cosine"
)



# ==== Trainer ====
trainer = SFTTrainer(
    args = training_args,
    model=model,
    train_dataset=train_dataset,
)

# ==== Train ====
trainer.train()

# ==== Save Model ====
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")



    



model.eval()

pipe = pipeline(
    "text-generation",
    model=output_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

#create embeddings
index = create_embeddings(train_df)
preds, actuals, texts, similar_tasks_records = [], [], [], []
#Loop through all test tasks
for i, test_task in test_df.iterrows():
    similar_tasks, sim_scores = retrieve_similar_tasks(index, train_df, test_task['text'])
    sim_tasks = []

    for i, (idx,score) in enumerate(zip(similar_tasks.index, sim_scores)):
        task_info = {
            "rank": i + 1,
            "similarity_score": float(score),
            "title": str(train_df.loc[idx, "title"]),
            "description": str(train_df.loc[idx, "summarized_description"]),
            "story_point": str(train_df.loc[idx, "storypoint"])
        }

        sim_tasks.append(task_info)
    
    # predicted = predict_story_point(model, tokenizer, sim_tasks, test_task, ft_model_id)
    predicted = generate(sim_tasks, test_task, pipe)


    preds.append(predicted)
    actuals.append(float(test_task["storypoint"]))
    texts.append(test_task["text"])
    similar_tasks_records.append(json.dumps(sim_tasks, ensure_ascii=False))

df_out = pd.DataFrame({
    "text": texts,
    "similar_tasks": similar_tasks_records,
    "actual_story_point": actuals,
    "predicted_story_point": preds
})

df_out.to_csv("Another fine-tuning/final_output.csv", index=False)

mae = mean_absolute_error(actuals, preds)
medae = median_absolute_error(actuals, preds)

print("MAE:", mae)
print("MDAE:",medae)