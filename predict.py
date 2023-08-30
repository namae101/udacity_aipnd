import argparse
import torch
from torchvision import transforms 
from PIL import Image
import json

def get_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('path_to_image', type=str, help='path to the image of flower')
    parser.add_argument('checkpoint', type=str, help='path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='number of top classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='path to the json file of category names')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    class_to_idx = checkpoint['class_to_idx']

    return model, class_to_idx

def process_image(image):
    with Image.open(image) as im:
        image_transformer = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
        return image_transformer(im)

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    input_img = process_image(image_path)
    input_img = input_img.view(1,3,224,224).to(device)
    logps = model(input_img)
    
    ps = torch.exp(logps)
    
    top_p, top_class = ps.topk(topk, dim=1)
    top_class = top_class.view(topk)
    top_p = top_p.view(topk)
    
    return top_class, top_p

def main():
    in_arg = get_input_args()

    #check if image file exist
    try:
        Image.open(in_arg.path_to_image)
    except:
        print('Please provide a valid path to the image file')
        return None

    # Load model
    try:
        model, class_to_idx = load_checkpoint(in_arg.checkpoint)
    except:
        print('Please provide a valid path to the model checkpoint')
        return None

    # Load category names
    try:
        cat_to_name = json.load(open(in_arg.category_names))
    except:
        print('Please provide a valid path to the json file of category names')
        return None

    # Us CPU or GPu
    device = torch.device('cpu')
    if in_arg.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            print('GPU is not available, use CPU instead for Prediction')
        else:
            print('GPU is available, Using GPU for Prediction')
    else:
        print('Using CPU for Prediction')

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    top_class, top_p = predict(in_arg.path_to_image, model, device, in_arg.top_k)

    actual_class = in_arg.path_to_image.split('/')[-2]
    print('Actual class: {}'.format(cat_to_name[actual_class]))
    print('Top {} classes:'.format(in_arg.top_k))
    for i in range(in_arg.top_k):
        print('Class: {}, Probability: {:.3f}'.format(cat_to_name[idx_to_class[top_class[i].item()]], top_p[i].item()))


if __name__ == "__main__":
    main()