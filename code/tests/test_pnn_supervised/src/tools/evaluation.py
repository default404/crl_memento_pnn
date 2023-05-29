import torch
from torch.autograd import Variable
from tqdm import tqdm


def evaluate_model(model, input_dict, x, y, dataset_loader, **kwargs):
    total = 0.
    correct = 0
    for images, labels in tqdm(dataset_loader, ascii=True):
        x.resize_(images.size()).copy_(images)
        y.resize_(labels.size()).copy_(labels)
        
        with torch.no_grad():
            # inputs = Variable(x.view(x.size(0), -1))
            input_dict['obs'] = x
            preds, _ = model(input_dict)
            input_dict["prev_action"] = preds

            _, predicted = torch.max(preds.data, 1)

            total += labels.size(0)
            correct += (predicted == y).sum()

    return correct / total