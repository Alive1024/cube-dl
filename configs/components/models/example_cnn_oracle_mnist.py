from config_sys import model_getter, shared_config
from models.example_cnn import ExampleCNN


@model_getter
def get_model_instance():
    return ExampleCNN(num_input_channels=1, num_classes=shared_config.get("num_classes"))
