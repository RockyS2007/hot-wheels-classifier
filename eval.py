import torch
from PIL import Image

def evaluation(model, dataloader, num_examples, criterion, device):

    model.eval()

    running_loss = 0.0
    running_corrects = 0

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)   # doing same thing as torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        eval_loss = running_loss / num_examples
        eval_accuracy = running_corrects.double() / num_examples

        print("Test Loss: {:.4f} Test Accuracy: {:.4f}".format(eval_loss, eval_accuracy))
        return eval_loss, eval_accuracy

def predict_image(model, class_names, image_path, transform, device):
    model.eval()

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    # basically the input to the model must take the shape of [batches_size, channels, height, width]
    # we must unsqueeze it to add a dummy batch_size of 1

    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)    # for the reasons above, we softmax dim 1

    confidence, index = probabilities.max(dim=1)

    predicted_class = class_names[index.item()]
    confidence = confidence.item()

    print(f'The model predicts this image is: {predicted_class} with {confidence * 100}% confidence')

    return predicted_class, confidence    
