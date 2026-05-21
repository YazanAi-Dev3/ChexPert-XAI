import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text_1 = """# XAI Evaluation Results Analysis
This notebook analyzes the XAI pipeline evaluation results and visualizes the performance across different models, explainers, thresholding techniques, and bounding box methods.
"""

code_1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Define paths
RESULTS_DIR = '.'
FINAL_RESULTS_PATH = f'{RESULTS_DIR}/final_test_results.csv'
VAL_COMBO_PATH = f'{RESULTS_DIR}/validation_combo_results.csv'
BEST_PIPELINES_PATH = f'{RESULTS_DIR}/best_pipelines.json'
"""

text_2 = """## 1. Validation Results Analysis (Thresholds & BBox Methods)
Here we plot the bounding box IoU for each Model+Explainer combinations. The x-axis represents the Threshold method, the y-axis is the BBox IoU, and the colors represent the BBox extraction method.
"""

code_2 = """# Load validation combo results
val_df = pd.read_csv(VAL_COMBO_PATH)

# The 'engine' column contains both model and explainer (e.g., ImageNet_GradCAM)
engines = val_df['engine'].unique()

# Calculate mean IoU for each combination
agg_df = val_df.groupby(['engine', 'threshold', 'bbox_method'])['bbox_iou'].mean().reset_index()

for engine in engines:
    engine_df = agg_df[agg_df['engine'] == engine]
    
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar chart
    sns.barplot(data=engine_df, x='threshold', y='bbox_iou', hue='bbox_method', palette='viridis')
    
    plt.title(f'BBox IoU by Threshold and BBox Method - {engine}')
    plt.xlabel('Threshold Method')
    plt.ylabel('Mean Bounding Box IoU')
    plt.xticks(rotation=45)
    plt.legend(title='BBox Method')
    plt.tight_layout()
    plt.show()
"""

text_3 = """## 2. Best Pipelines Overview
Summary of the best pipelines chosen from the validation phase.
"""

code_3 = """with open(BEST_PIPELINES_PATH, 'r') as f:
    best_pipelines = json.load(f)

best_df = pd.DataFrame(best_pipelines).T
display(best_df)
"""

text_4 = """## 3. Final Test Set Results
Comparing the top pipelines on the unseen test set.
"""

code_4 = """# Load final test results
test_df = pd.read_csv(FINAL_RESULTS_PATH)

# Aggregate test results by engine
test_agg = test_df.groupby('engine').agg({
    'bbox_iou': 'mean',
    'pixel_iou': 'mean',
    'point_game': 'mean',
    'recall': 'mean',
    'area_ratio': 'mean'
}).reset_index()

# Plot BBox IoU on Test Set
plt.figure(figsize=(10, 6))
sns.barplot(data=test_agg, x='engine', y='bbox_iou', palette='Set2')
plt.title('Mean BBox IoU on Final Test Set')
plt.xlabel('Engine (Model + Explainer)')
plt.ylabel('Mean BBox IoU')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

display(test_agg)
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(text_4),
    nbf.v4.new_code_cell(code_4)
]

with open('d:/Work-M/Master-CORAX/xai_results/analyze_results.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook successfully created at d:/Work-M/Master-CORAX/xai_results/analyze_results.ipynb")
