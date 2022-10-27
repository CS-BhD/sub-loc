import time
import torch
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

def train(train_iter, net, loss, optimizer, num_epochs):
    start_time = time.time()
    for epoch in range(num_epochs):
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

            if i % 100 == 99:
                print("epoch {}, iteration {} , loss {:.3f}, time {:.3f} sec".format(epoch, i, l.item(), time.time() - start_time))
    print("Finshed Train !")


# 训练集上的表现
def evaluate_model(model, dataloder):
    preds = []
    y_true = []
    correct = 0 
    total = 0
    with torch.no_grad():
        for sequences, labels in dataloder:
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            preds.extend(predicted.tolist())
            y_true.extend(labels.tolist())
    print('Accuracy of the network on the train set %d %%' % (100 * correct / total))
    cm = confusion_matrix(y_true, preds)
    # cm_list.append(cm)
    # drawing_tools.plot_confusion_matrix(cm, classes)
    mcm = multilabel_confusion_matrix(y_true, preds)
    # mcm_list.append(mcm)
    return cm, mcm