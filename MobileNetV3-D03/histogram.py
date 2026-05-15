import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load test results
with open('/home/drive3/10k_MobileNet_tsm/MobileNetV3-D03/test_results.json', 'r') as f:
    results = json.load(f)

f1_scores = results['Per-flag F1 scores']

# Prepare data for plotting
flags = [
    'Spatial', 
    'Hallucination', 
    'Lighting', 
    'Rendering', 
    'Physics', 
    'Flicker', 
    'Motion'
]

# Map the raw keys to the readable flags above
keys = [
    'spatial_flag_f1', 
    'hallucination_flag_f1', 
    'lighting_flag_f1', 
    'rendering_flag_f1', 
    'physical_violation_flag_f1', 
    'object_flicker_flag_f1', 
    'motion_inconsistency_flag_f1'
]

scores = [f1_scores.get(key, 0) for key in keys]

# Create figure
plt.figure(figsize=(10, 6))

# Choose colors for bars
colors = ['#58a6ff', '#39d353', '#f0883e', '#d2a8ff', '#ff7b72', '#a5d6ff', '#e3b341']

# Plot bars
bars = plt.bar(flags, scores, color=colors, edgecolor='black', alpha=0.8)

# Set titles and labels
plt.title('MobileNetV3-TSM (Dropout=0.3): F1 Score vs Artifact Flags', fontsize=14, pad=15)
plt.xlabel('Artifact Flags', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)

# Set y-axis limit to 0.7 
plt.ylim(0, 0.7)

# Add exact value annotations on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Grid
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save to project dir and artifacts dir
out_path1 = '/home/drive3/10k_MobileNet_tsm/MobileNetV3-D03/f1_histogram.png'
out_path2 = '/home/user3/.gemini/antigravity/brain/7c3563ce-f275-4408-bf62-b4bb5472bef5/f1_histogram_v3.png'

plt.savefig(out_path1, dpi=300)
plt.savefig(out_path2, dpi=300)
plt.close()

print(f"F1 Histogram successfully generated at {out_path1}")