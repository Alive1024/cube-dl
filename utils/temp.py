import torch


def inspect_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    for key, value in ckpt.items():
        if isinstance(value, dict):
            print(key, value.keys())
        else:
            print(key, value)


if __name__ == '__main__':
    inspect_ckpt("//src/outputs/Image-Classification"
                 "-on-Oracle-MNIST/run_14/fit/checkpoints/epoch=2-step=1149.ckpt")
