import torch

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models_train/petom_weights.pt', force_reload=True)
    model.eval()
    return model
