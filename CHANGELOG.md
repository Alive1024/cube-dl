# v0.3.8
- In task modules, we do not need to call the model getter(s) explicitly to get the model object(s) and use it in optimizer, etc. Instead, task module getters now are compulsorily required to declare a parameter named `model` to accept the model object(s), which will be instantiated by the root config.
