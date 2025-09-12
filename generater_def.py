import torch
import pandas as pd
from transformers import pipeline

# ---- Load model ----
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    # device = 0
    device_map="auto",
)

def generate(similar_tasks, test_task):
    formatted_similar_tasks = "\n\n".join([
        f"Task {task['rank']} (Similarity: {task['similarity_score']:.4f}):\n"
        f"Title: {task['title']}\n"
        f"Description: {task['description']}\n"
        f"Story Point: {task['story_point']}\n\n"
        for task in similar_tasks
    ])

    # prompt = f'''You are an expert Agile story point estimator. I will give you a new task along with 3 similar tasks and their story points.
    # Your job is to estimate the story point for the new task based on these references.'''

    prompt = f'''You are an expert Agile story point estimator.
    I will give you a new task for which you have to estimate the story point. Story points are used to estimate the overall effort required to complete a task. They are chosen from a predefined scale, often using the Fibonacci sequence - 0, 1, 2, 3, 5, 8 and so on. Story points do not have fractional values.
    For each new task, I will provide 3 similar tasks which already have story points recorded. While you try to estimate the story point for the new task, you can use these as reference because these tasks are from the same project. 
    Your job is to estimate the story point after considering the complexity of the new task. You can use the provided similar tasks for further reference.'''

    user_content = f'''
    New Task:
    Title: {test_task["title"]}
    Description: {test_task["summarized_description"]}

    Similar Tasks:
    {formatted_similar_tasks}

    Please answer with only your estimated story point for the new task. Your output should be in this form: "story point: estimated story point".
    '''

    # print(user_content)



    messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": user_content},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=256,
        temperature=0.2,
    )

    sp = outputs[0]["generated_text"][-1]["content"]

    return sp
