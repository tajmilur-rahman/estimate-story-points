import pandas as pd
from sklearn.model_selection import train_test_split
import json





# Helper to convert rows to instruction format
def format_row(row):
    ############################fibonacci numbers mentioned
    system_prompt = f'''You are an expert Agile story point estimator.
    I will give you a new task for which you have to estimate the story point. Story points are used to estimate the overall effort required to complete a task. They are chosen from a predefined scale, often using the Fibonacci sequence - 0, 1, 2, 3, 5, 8 and so on. Story points do not have fractional values. 
    Your job is to estimate the story point after considering the complexity of the new task.'''

# ################################# fibonacci numbers not mentioned
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
    # '''
################################################simple prompt
    # return {
    #     "messages": [
    #         {
    #             "role": "system",
    #             "content" : system_prompt

    #         },
    #         {
    #             "role": "user",
    #             "content": f'''
    #                         New Task:
    #                         {row['text']}
    #                         '''
    #         },
    #         {
    #             "role": "assistant",
    #             "content": f'''story point: {str(row['storypoint'])}'''
    #         }
    #     ]
    # }


    return {
        "messages": [
            {
                "role": "system",
                "content" : system_prompt

            },
            {
                "role": "user",
                "content": f'''
                            New Task:
                            {row['text']}
                            Please answer with only your estimated story point for the new task. Your output should be in this form: "story point: estimated story point".
                            '''
            },
            {
                "role": "assistant",
                "content": f'''story point: {str(row['storypoint'])}'''
            }
        ]
    }

def format_data(summarized_file):
    df = pd.read_csv(summarized_file)
    df["text"] = "Task title: " + df["title"].fillna('') + "\nTask Description:\n" + df["summarized_description"].fillna('')

    n = len(df)
    train_end = int(0.8 * n)
    # Split
    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end:]


    # Apply formatting
    train_data = train_df.apply(format_row, axis=1).tolist()
    test_data = test_df.apply(format_row, axis=1).tolist()

    # Save as JSONL
    with open("/u1/users/fnv012/SPLLAMA3/Another fine-tuning/train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open("/u1/users/fnv012/SPLLAMA3/Another fine-tuning/test.jsonl", "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    return train_df, test_df
