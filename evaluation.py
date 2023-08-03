import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

model = load_model("C:/Users/Anant/Desktop/New folder/mobilenet_fine_tuned_model.h5")  
test_data_dir = "C:/Users/Anant/Desktop/New folder/test"
img_size = (224, 224)
batch_size = 32
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=img_size, batch_size=batch_size,
                                                  class_mode='categorical', shuffle=False)


y_true = test_generator.classes  
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)  

report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())
confusion = confusion_matrix(y_true, y_pred)

print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion)

def extract_metrics(report):
    lines = report.split('\n')
    metrics = {}
    for line in lines[2:]:
        cols = line.split()
        if len(cols) < 5:
            continue
        class_name = ' '.join(cols[:-4])
        precision = float(cols[-4])
        recall = float(cols[-3])
        f1_score = float(cols[-2])
        metrics[class_name] = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
    return metrics

metrics = extract_metrics(report)

class_names = list(metrics.keys())
precision_values = [metrics[class_name]['precision'] for class_name in class_names]
recall_values = [metrics[class_name]['recall'] for class_name in class_names]
f1_score_values = [metrics[class_name]['f1_score'] for class_name in class_names]

for class_name, precision, recall, f1_score in zip(class_names, precision_values, recall_values, f1_score_values):
    print(f"Class: {class_name}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
    print("="*40)


