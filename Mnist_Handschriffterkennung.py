#*************************************************************************************************************#
#Autor: Martin Bischof                                                                                        #
#Erstellt am : 04.02.2020                                                                                     #
#Kredits: https://www.youtube.com/watch?v=8gZR4Q3262k&list=PLNmsVeXQZj7rx55Mai21reZtd_8m-qe27&index=10        #
#Wie installiere ich PyTorch?: www.bischofmartin.de/pytorch.php                                               #
#Quellcode erklärt: www.bischofmartin.de/handschriffterkennung.php                                            #
#Link zu GitHub Repo: https://github.com/MartinBischof/Handschriffterkennung_MNIST                            #
#Hinweis: Die Datei Mnist_Handschriffterkennung.pyproj ist für die eigentliche Funktion nicht nöttig          #
#*************************************************************************************************************#
import torch 
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as o
from torch.autograd import Variable
from torchvision import datasets, transforms 
from PIL import Image 
import os 
kwargs = {'num_workers': 0, 'pin_memory': False}
#print(torch.cuda.is_available())
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
    ) #Ende Daten_Test



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
        x = f.relu(self.Layer4_fullyConected(x))
        x = self.Layer5_output(x)
        return f.log_softmax(x,dim=1)

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
        ErrorValue = ErrorFunk(ergebniss,target)
        ErrorValue.backward()
        optimizer.step()
        #Ausgabe
        #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #    epoch, batch_id*len(data), len(Daten_Training.dataset),
        #    100.*batch_id/len(Daten_Training), ErrorValue.item()))

def test():
    Netz.eval()
    ErrorValue = 0
    korekt = 0
    for data, target in Daten_Test:
        #print(data)
        with torch.no_grad():
            data = Variable(data.cuda())
        target = Variable(target.cuda())
        ergebnis = Netz(data)
        ErrorValue += f.nll_loss(ergebnis, target, size_average=False).item()
        prediction = ergebnis.data.max(1, keepdim = True)[1]
        korekt += prediction.eq(target.data.view_as(prediction)).cpu().sum()
    #print('Durchschnittserror: ', ErrorValue/len(Daten_Test.dataset))
    #print('Genauigkeit', 100.*korekt/len(Daten_Test.dataset))
    #print(prediction)

#parameter ist ein string also in 'eins.png'
def guess(imgName):
    Netz.eval()
    img = Image.open(imgName)
    img = img.resize((28, 28), Image.BILINEAR)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1370,),(0.3081,))])
    img_tensor = transform(img).float()
    with torch.no_grad():
        data = Variable(img_tensor)
    data = data.unsqueeze(1)
    #print(data)
    data = data.cuda()
    ergebnis = Netz(data)

    print(ergebnis.data.max(1, keepdim = True)[1]) #ergebnis ausgeben
    #print(ergebnis.data)


if __name__ == '__main__':
    if os.path.isfile('handschriftserkennung.pt'):
        Netz = torch.load('handschriftserkennung.pt')
    #for epoch in range(1,1):
    #    torch.save(Netz,'handschriftserkennung.pt');
    #    train(epoch)
    #    test()
    guess("sieben.png")

    
