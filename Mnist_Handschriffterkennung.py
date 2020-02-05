#*************************************************************************************************************#
#Autor: Martin Bischof                                                                                        #
#Erstellt am : 04.02.2020                                                                                     #
#Kredits: https://www.youtube.com/watch?v=8gZR4Q3262k&list=PLNmsVeXQZj7rx55Mai21reZtd_8m-qe27&index=10        #
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
    datasets.MNIST('data',train=False, download=True, 
                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1370,),(0.3081,))])),
    batch_size=512, #nicht zu groß setzen wenn man schlechte Hardware besitzt (~64)
    shuffle = True,
    **kwargs
    ) #Ende Daten_Training
#Das Neuronale Netzwerk (Model):
class NeuronalesNetwerk (nn.Module):
    def __init__(self):
        super(NeuronalesNetwerk,self).__init__()
        self.Layer1_input = nn.Conv2d(1,10,kernel_size=5)
        self.Layer2_convolutional = nn.Conv2d(10,20,kernel_size=5)
        self.Layer3_dropout = nn.Dropout2d()
        self.Layer4_fullyConected = nn.Linear(320,70)
        self.Layer5_output = nn.Linear(70,10)
        
    def forward(self,x):
        x = self.Layer1_input(x)
        x = f.max_pool2d(x,2)
        x = f.relu(x)
        x = self.Layer2_convolutional(x)
        x = self.Layer3_dropout(x)
        x = f.max_pool2d(x,2)
        x = f.relu(x)
        x = x.view(-1,320)
        x = self.Layer4_fullyConected(x)
        x = f.relu(x)
        x = self.Layer5_output(x)
        return f.log_softmax(x)

Netz = NeuronalesNetwerk()
Netz = Netz.cuda()

#Optimizer (Berechnet die Error-Funktion):
optimizer = o.SGD(Netz.parameters(), lr=0.1, momentum=0.8)

#Zusatzfunktionen:
def train(epoch):
    Netz.train()
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
        #Ausgabe
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id*len(data), len(Daten_Training.dataset),100.*batch_id/len(Daten_Training),
            ErrorValue.data[0]))

for epoch in range(1,30):
    train(epoch)
