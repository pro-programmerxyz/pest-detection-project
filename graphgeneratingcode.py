import matplotlib.pyplot as plt


class_names = ['paddy with pest', 'paddy without pest']
precision_values = [1.0, 1.0]
recall_values = [1.0, 1.0]
f1_score_values = [1.0, 1.0]
fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(class_names))


width = 0.2
ax.bar(x, precision_values, width=width, label='Precision')
ax.bar([i + width for i in x], recall_values, width=width, label='Recall')
ax.bar([i + 2 * width for i in x], f1_score_values, width=width, label='F1 Score')
ax.set_xticks([i + width for i in x])
ax.set_xticklabels(class_names)
ax.set_ylabel('Score')
ax.legend()
plt.tight_layout()
plt.show()
