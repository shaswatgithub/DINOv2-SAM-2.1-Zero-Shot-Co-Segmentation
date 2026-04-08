import json

# Load the notebook
with open('/home/shaswat22/Downloads/llama3_1_sentiment_pyspark_project.ipynb', 'r') as f:
    nb = json.load(f)

# Remove widgets from metadata
if 'widgets' in nb['metadata']:
    del nb['metadata']['widgets']

# Remove widget outputs from cells
for cell in nb['cells']:
    if 'outputs' in cell:
        cell['outputs'] = []

# Save the notebook
with open('/home/shaswat22/Downloads/llama3_1_sentiment_pyspark_project.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)