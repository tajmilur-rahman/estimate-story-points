from retriever_def import *
from generator_def import *
from sklearn.metrics import mean_absolute_error, median_absolute_error
import numpy as np




# Load dataset
train_df, test_df = load_and_prepare_data("summarized_aloy2.csv")

#create embeddings
index = create_embeddings(train_df)


predictions = []
actual = []
#Loop through all test tasks
for i, test_task in test_df.iterrows():
    similar_tasks, sim_scores = retrieve_similar_tasks(index, train_df, test_task['text'])
    sim_tasks = []

    for i, (idx, score) in enumerate(zip(similar_tasks.index, sim_scores)):
        task_info = {
            "rank": i + 1,
            "similarity_score": float(score),
            "title": train_df.loc[idx, "title"],
            "description": train_df.loc[idx, "summarized_description"],
            "story_point": train_df.loc[idx, "storypoint"]
        }

        sim_tasks.append(task_info)
        

        # print(f"\nTop {i+1} Similar Task (Score: {score:.4f}):")
        # print("Title:", train_df.loc[idx, "title"])
        # print("Description:", train_df.loc[idx, "summarized_description"])
        # print("Story Point:", train_df.loc[idx, "storypoint"])

    sp = generate(sim_tasks, test_task)

    # Try to extract the numeric value (if LLM outputs text like "story point: 3")
    try:
        sp_num = int(''.join(filter(str.isdigit, str(sp))))
    except:
        sp_num = None
    predictions.append(sp_num)
    
    actual.append(task_info["story_point"])

mae = mean_absolute_error(actual, predictions)
medae = median_absolute_error(actual, predictions)

print("\nEvaluation Results:")
print("Mean Absolute Error:", mae)
print("Median Absolute Error:", medae)
