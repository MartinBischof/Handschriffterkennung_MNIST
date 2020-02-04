#*************************************************************************************************************#
#Autor: Martin Bischof                                                                                        #
#(Kredits: https://www.youtube.com/watch?v=8gZR4Q3262k&list=PLNmsVeXQZj7rx55Mai21reZtd_8m-qe27&index=10)      #
#Wie installiere ich PyTorch?: www.bischofmartin.de/pytorch.php                                               #
#Link zu GitHub Repo: https://github.com/MartinBischof/Handschriffterkennung_MNIST                            #
#Erstellt am : 04.02.2020                                                                                     #
#*************************************************************************************************************#


import torch 
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as o
from torch.autograd import Variable
from torchvision import datasets, transforms 
kwargs = {'num_workers': 1, 'pin_memory': True}

Daten_Training = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=True, download=True, 
                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1370,),(0.3081,))])),
    batch_size=512, #nicht zu groß setzen wenn man schlechte Hardware besitzt (~64)
    shuffle = True,
    **kwargs
    ) #Ende Daten_Training

Daten_Test = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=True, download=True, 
                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1370,),(0.3081,))])),
    batch_size=512, #nicht zu groß setzen wenn man schlechte Hardware besitzt (~64)
    shuffle = True,
    **kwargs
    ) #Ende Daten_Training

