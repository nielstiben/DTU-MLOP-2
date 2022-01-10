import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Supress Tensorboard pollution.
import torchvision.transforms as transforms

from torchvision.datasets import MNIST

dataset_path = "datasets"
mnist_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)
assert len(train_dataset) == 60000
assert len(test_dataset) == 10000