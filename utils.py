import torch

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)  # Получаем индексы максимальных значений
    correct = (preds == labels).sum().item()  # Считаем количество правильных предсказаний
    accuracy = correct / labels.numel()  # Общее количество элементов в маске
    return accuracy
