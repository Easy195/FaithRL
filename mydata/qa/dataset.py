import pandas as pd
import json
from pathlib import Path

input_json_path = Path("xxx.json")
output_path = Path("xxx.json")

with input_json_path.open("r", encoding="utf-8") as f:
    input_data = json.load(f)

rows = input_data
out = []

for idx, row in enumerate(rows):
    sample_id = idx
    knowledge = row.get("knowledge", "")
    question = row.get("question", "")
    right_answer = row.get("answer", "")

    few_shot_prompt = """Use the following knowledge to answer the given question accurately and only based on the knowledge provided.
Knowledge:
{knowledge}
Question:
{question}
Your answer MUST be enclosed in a LaTeX box like this: \\boxed{{your answer here}}.

Answer:""".format(question=question, knowledge=knowledge)

    out.append({
        "sample_id": sample_id,
        "prompt": [
            {"role": "user", "content": few_shot_prompt}
        ],
        "extra": [
            {
                "context": knowledge,
                "gold_answer": right_answer,
                "question": question
            }
        ]
    })

with output_path.open("w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

json_file = "final.json"
df = pd.read_json(json_file)
train_df = df
train_df.to_parquet("mydata/qa/train.parquet", index=False)

