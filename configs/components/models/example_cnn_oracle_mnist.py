from models.example_cnn import ExampleCNN


def get_model_instance():
    return ExampleCNN(num_input_channels=1, num_classes=10)
