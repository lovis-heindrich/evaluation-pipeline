import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, brier_score_loss


def train(network, optimizer, loss_fn, epochs, trainloader, testloader, device, log = False):
    accuracies = []
    for epoch in range(epochs):
        network.train()
        for i, data in enumerate(trainloader):
            x, y = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            output = network(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

        network.eval()
        correct = 0
        total = 0
        
        for i, data in enumerate(testloader):
            x, y = data[0].to(device), data[1].to(device)
            
            with torch.no_grad():
                output = network(x)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == y).sum().item()
                total += y.shape[0]
        accuracies.append(correct/total)
        if log and epoch+1 is not epochs:
            print('Epoch', epoch+1, 'Accuracy:', accuracies[-1])
    if log:
        print('Epoch', epochs, 'Accuracy:', accuracies[-1])
    return accuracies

def test(network, testset, device, num_classes = 14):
    total = 0.0
    correct = 0.0
    activations = []
    activations_correct = []
    activations_incorrect = []
    y_true = []
    y_pred = []
    outputs = []
    for i, data in enumerate(testset):
        x, y = data[0].to(device), data[1].to(device)
        
        with torch.no_grad():
            output, activation, predicted = network.get_prediction(x)  
            outputs.append(output)
            y_true.append(y)
            y_pred.append(predicted)         
            correct += (predicted == y).sum().item()
            total += y.shape[0]
            activations.append(activation)
            activations_correct.append(activation[predicted==y])
            activations_incorrect.append(activation[predicted!=y])
    
    activations_correct_tensor = torch.cat(activations_correct).to("cpu")
    activations_incorrect_tensor = torch.cat(activations_incorrect).to("cpu")

    std_activation, average_activation = torch.std_mean(torch.cat(activations).to("cpu"))
    std_activation_correct, average_activation_correct = torch.std_mean(torch.cat(activations_correct).to("cpu"))
    std_activation_incorrect, average_activation_incorrect = torch.std_mean(torch.cat(activations_incorrect).to("cpu"))
    y_true = torch.cat(y_true).to("cpu").numpy()
    y_pred = torch.cat(y_pred).to("cpu").numpy()
    matrix = confusion_matrix(y_true, y_pred)
    correct_vector = (y_true == y_pred)
    roc = roc_auc_score(correct_vector, torch.cat(activations).to("cpu").numpy())
    pr_in_score = average_precision_score(correct_vector, torch.cat(activations).to("cpu").numpy())
    pr_out_score = average_precision_score(1- correct_vector, -1 * torch.cat(activations).to("cpu").numpy())

    # Brier loss
    activations_brier = torch.cat(outputs)
    labels = torch.Tensor(y_true).long()
    y_brier = torch.nn.functional.one_hot(labels, num_classes).float().to(device)
    brier = torch.nn.MSELoss()
    brier_loss = brier(activations_brier, y_brier)

    metrics = {
        "activation_mean": average_activation,
        "activation_std": std_activation,
        "activation_correct_mean": average_activation_correct,
        "activation_correct_std": std_activation_correct,
        "activation_incorrect_mean": average_activation_incorrect,
        "activation_incorrect_std": std_activation_incorrect,
        "confusion_matrix": matrix,
        "n": total,
        "n_correct": correct,
        "n_incorrect": (total-correct),
        "activations_correct": activations_correct_tensor,
        "activations_incorrect": activations_incorrect_tensor,
        "roc": roc,
        "pr_in": pr_in_score,
        "pr_out": pr_out_score,
        "brier": brier_loss.to("cpu").item()
    }
    return correct/total, average_activation, metrics