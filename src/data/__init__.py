from .cifar10 import Cifar10, Cifar10C
from .cifar100 import Cifar100, Cifar100C

class Cifar10CData:
    name = "Cifar10C"
    source = Cifar10
    data = Cifar10C
    n_classes = 10

class Cifar100CData:
    name = "Cifar100C"
    source = Cifar100
    data = Cifar100C
    n_classes = 100
