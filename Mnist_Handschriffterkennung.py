#*************************************************************************************************************#
#Autor: Martin Bischof                                                                                        #
#Erstellt am : 04.02.2020                                                                                     #
#(Kredits: https://www.youtube.com/watch?v=8gZR4Q3262k&list=PLNmsVeXQZj7rx55Mai21reZtd_8m-qe27&index=10)      #
#Wie installiere ich PyTorch?: www.bischofmartin.de/pytorch.php                                               #
#Link zu GitHub Repo: https://github.com/MartinBischof/Handschriffterkennung_MNIST                            #
#Hinweis: Die Datei Mnist_Handschriffterkennung.pyproj ist für die eigentliche Funktion nicht nöttig          #
#*************************************************************************************************************#


import torch 
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as o
from torch.autograd import Variable
from torchvision import datasets, transforms 
kwargs = {'num_workers': 1, 'pin_memory': True}

#Daten:
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
#Das Neuronale Netzwerk (Model):
class NeuronalesNetwerk (nn.Module):
    def __init__(self):
        pass
    def forward(self,x):
        pass

Netz = NeuronalesNetwerk()
Netz = Netz.cuda()

#Optimizer (Berechnet die Error-Funktion):
optimizer = o.SGD(Netz.parameters(), lr=0.1, momentum=0.8)

#Zusatzfunktionen:
def train(epoch):
    model.train()
    for batch_id, (data,target) in enumerate(Daten_Training):
        data = data.cuda()
        target = target.cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        ergebniss = Netz(data)
        ErrorFunk = f.nll_loss
        ErrorValue = ErrorFunk(out,target)
        ErrorValue.backwards()
        optimizer.step()
        #print('')

for epoch in range(1,30):
    train(epoch)
