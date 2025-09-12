import torch
import pandas as pd
from transformers import pipeline



def summarize_csv(input_csv_path, output_csv_path = "summarized.csv"):


    # Load model
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    #Load CSV file
    df = pd.read_csv(input_csv_path)


    #Summarize each description
    summaries = []
    for desc in df['description']:
        if pd.isna(desc) or not str(desc).strip():
            summaries.append("")  # Keep empty if no description
            continue
        
        messages = [
            {"role": "system", "content": "Your are a helpful assistant who only summarizes the given content. You will be given description of a task from a project and you have to summarize the task description."},
            {"role": "user", "content": f"Your answer should contain only the summary. Summarize this task description : '''{str(desc)}'''"},
        ]
        
        outputs = pipe(
            messages,
            max_new_tokens=256,
        )
        
        
        summary = outputs[0]["generated_text"][-1]["content"]
        summaries.append(summary)
        


    df['summarized_description'] = summaries


    
    df.to_csv(output_csv_path, index=False)

    print(f"Summarized CSV saved as: {output_csv_path}")
    return output_csv_path
