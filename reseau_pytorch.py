import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
import os
from random import randint
from PIL import Image
from ast import literal_eval
os.chdir("/Reseau/drive/MyDrive/Reseau_neuronne_pyTorch")
from image_transformer import create_data, delete_data
from fonctions import sigmoide,tangente

def creation_training_data(cercle,rotation,taille):
  delete_data()
  create_data(cercle,rotation,taille)




Nimage=28
Ncouche1=512
Ncouche2=512
Ncouche3=10

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() #initialise proprement le module nn
        self.flatten = nn.Flatten()  #Définit self.flatten(), une fonction applatissant un tableau classique en un tensor
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(Nimage*Nimage,Ncouche1), #Crée une fonction prenant en argument deux tailles n et applique
                                                           #un jeu d'entrée à une couche d'entrée de taille n et une de sortie de taille m
            nn.ReLU(),                      #Affiche le max entre 0 et l'ouput de l'instruction précédente
            nn.Linear(Ncouche1, Ncouche2),
            nn.ReLU(),
            nn.Linear(Ncouche2, Ncouche3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

input_image = torch.rand(3,28,28)
print(input_image.size())

#https://pytorch.org/vision/stable/generated/torchvision.datasets.GTSRB.html#torchvision.datasets.GTSRB
#https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
#https://pytorch.org/docs/stable/nn.html
#https://www.v7labs.com/open-datasets/gtsrb