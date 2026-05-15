import json
import matplotlib.pyplot as plt

# Load test results
with open('/home/drive3/10k_MobileNet_tsm/test_results.json', 'r') as f:
    results = json.load(f)

f1_scores = results['Per-flag F1 scores']

flags = ['Spatial', 'Hallucination', 'Lighting', 'Rendering', 'Physics', 'Motion', 'Flicker']
values = [
    f1_scores['spatial_flag_f1'],
    f1_scores['hallucination_flag_f1'],
    f1_scores['lighting_flag_f1'],
    f1_scores['rendering_flag_f1'],
    f1_scores['physical_violation_flag_f1'],
    f1_scores['motion_inconsistency_flag_f1'],
    f1_scores['object_flicker_flag_f1']
]

# Create figure
plt.figure(figsize=(10, 6))
bars = plt.bar(flags, values, color='#1f77b4', edgecolor='white')

# Set titles and labels
plt.title('F1 Score vs Artifact Flags', fontsize=14)
plt.xlabel('Artifact Flags', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)

# Set y-axis limit to 0.7 exactly like the reference image
plt.ylim(0, 0.7)

# Add exact value annotations on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# Save to both project dir and artifacts dir
plt.savefig('/home/drive3/10k_MobileNet_tsm/f1_histogram.png', dpi=300)
plt.savefig('/home/user3/.gemini/antigravity/brain/7c3563ce-f275-4408-bf62-b4bb5472bef5/f1_histogram.png', dpi=300)
print("F1 Histogram successfully generated.")
