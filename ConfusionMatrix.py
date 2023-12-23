import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, accuracy,precision,recall,f1, title='Confusion Matrix'):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt="d", linewidths=.5, cmap="Blues")
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    metrics_string = (f'Accuracy: {accuracy:.4f} | '
                      f'Precision: {precision:.4f} | '
                      f'Recall: {recall:.4f} | '
                      f'F1-Score: {f1:.4f}')
    
    # Define a posição do texto, por exemplo, x será 0.5 (centro do gráfico)
    # e y será -0.1 (abaixo do eixo x do gráfico).
    # Ajuste esses valores conforme necessário para a posição correta.
    plt.text(0.5, -0.1, metrics_string, ha='center', va='top', transform=plt.gca().transAxes)
    
    plt.show()
