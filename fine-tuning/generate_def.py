import re

#########################inference
def generate(similar_tasks, test_task, pipe):
    formatted_similar_tasks = "\n\n".join([
        f"Task {task['rank']}:\n"
        f"Task Title: {task['title']}\n"
        f"Task Description: {task['description']}\n"
        f"Story Point: {task['story_point']}\n\n"
        for task in similar_tasks
    ])

    ############################fibonacci numbers mentioned
    system_prompt = f'''You are an expert Agile story point estimator.
    I will give you a new task for which you have to estimate the story point. Story points are used to estimate the overall effort required to complete a task. They are chosen from a predefined scale, often using the Fibonacci sequence - 0, 1, 2, 3, 5, 8 and so on. Story points do not have fractional values. 
    Your job is to estimate the story point after considering the complexity of the new task.'''

# ################################## fibonacci numbers not mentioned
#     system_prompt = f'''You are an expert Agile story point estimator.
#     I will give you a new task for which you have to estimate the story point. Story points are used to estimate the overall effort required to complete a task. They are chosen from a predefined scale, often using the Fibonacci sequence. Story points do not have fractional values. 
#     Your job is to estimate the story point after considering the complexity of the new task.'''

# ################################### previous prompt used
#     system_prompt = f'''Assume that you are a software developer for a project called ‘Compass’ (Repository: MongoDB, Project key: 'COMPASS'). This project follows Agile Software Development methodology.
#     In Agile project management, tasks are associated with ‘story points’. Story points indicate how complex and effort consuming a task is. The larger the story point, the more time and effort that task will require to be completed.
    
#     Your job as a software developer is to estimate the story point for a new task. Please answer only with the story point in integer.'''



#########################################simple prompt
    # system_prompt = f'''You are an expert Agile story point estimator.
    # Story points are used to estimate the overall effort required to complete a task. They are chosen from a predefined scale, often using the Fibonacci sequence - 0, 1, 2, 3, 5, 8 and so on. Story points do not have fractional values. 
    # Reply strictly as: story point: <integer>
    # Your job is to estimate the story point after considering the complexity of the new task.
    # You may use the similar tasks as guidance'''



    user_content = f'''For each new task, I will provide 3 similar tasks which already have story points recorded. While you try to estimate the story point for the new task, you can use these as reference because these tasks are from the same project. 
    Your job is to estimate the story point after considering the complexity of the new task. You can use the provided similar tasks for further reference.'''

    user_content += f'''
    New Task:
    Task Title: {test_task["title"]}
    Task Description: {test_task["summarized_description"]}

    List of similar tasks:
    {formatted_similar_tasks}

    Please answer with only your estimated story point for the new task. Your output should be in this form: "story point: estimated story point".
    '''


###############################################simple prompt
    # user_content = f'''
    # New Task:
    # Task Title: {test_task["title"]}
    # Task Description: {test_task["summarized_description"]}

    # List of similar tasks:
    # {formatted_similar_tasks}
    # '''

    
    # Create chat-based message structure
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Generate output with the fine-tuned model
    outputs = pipe(
        messages,
        max_new_tokens=20,  # Adjust max tokens as necessary
        temperature=0.3,     # Lower temperature for more deterministic outputs
    )

    # Extract generated story point from output
    sp = outputs[0]["generated_text"][-1]["content"]
    try:
        match = re.search(r"\d+(\.\d+)?", str(sp))  
        if match:
            sp_num = float(match.group()) 
        else:
            print("hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
            print(sp)
            sp_num = None
    except:
        print("hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        print(sp)
        sp_num = None

    return sp_num