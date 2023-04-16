from config_decorators import model_getter
from models.example_cnn import ExampleCNN


@model_getter
def get_model_instance():
    return ExampleCNN(num_input_channels=1, num_classes=10)
