from retriever_def import *
from generator_def import *
from summarizer_def import *
from sklearn.metrics import mean_absolute_error, median_absolute_error
import numpy as np
import json
import re


#summarize descriptions
summarized_file = summarize_csv("DM_deep-se.csv")

# Load dataset
train_df, test_df = load_and_prepare_data(summarized_file)

#create embeddings
index = create_embeddings(train_df)


predictions = []
actual = []
similar_tasks_records = []
results = []
#Loop through all test tasks
for i, test_task in test_df.iterrows():
    similar_tasks, sim_scores = retrieve_similar_tasks(index, train_df, test_task['text'])
    sim_tasks = []

    for i, (idx, score) in enumerate(zip(similar_tasks.index, sim_scores)):
        task_info = {
            "rank": i + 1,
            "similarity_score": float(score),
            "title": str(train_df.loc[idx, "title"]),
            "description": str(train_df.loc[idx, "summarized_description"]),
            "story_point": str(train_df.loc[idx, "storypoint"])
        }

        sim_tasks.append(task_info)
        

    sp = generate(sim_tasks, test_task)
    results.append(sp)

    try:
        match = re.search(r"\d+(\.\d+)?", str(sp))  
        if match:
            sp_num = float(match.group()) 
        else:
            sp_num = None
    except:
        sp_num = None

    predictions.append(sp_num)
    actual.append(test_task["storypoint"])
    similar_tasks_records.append(json.dumps(sim_tasks, ensure_ascii=False))
    


# add the actual sp and predicted sp to the dataset
test_df["actual_sp"] = actual
test_df["predicted_sp"] =  predictions
test_df["results"] = results
test_df["similar_tasks"] = similar_tasks_records
test_df.to_csv("predictions.csv", index=False)



mae = mean_absolute_error(actual, predictions)
medae = median_absolute_error(actual, predictions)

print("\nEvaluation Results:")
print("Mean Absolute Error:", mae)
print("Median Absolute Error:", medae)
