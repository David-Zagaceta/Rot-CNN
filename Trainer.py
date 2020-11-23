import torch
import numpy as np
import logging

class Trainer:

    def __init__(self, model, optimizer, lossfn, trainingdata, validationdata=None, validationinterval=1):
        self.model = model
        self.optimizer = optimizer
        self.lossfn = lossfn
        self.trainingdata = trainingdata
        self.validationdata = validationdata
        self.validationinterval = validationinterval

    def train(self, epochs):
        logging.basicConfig(filename='training.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
        logging.info("Training")
        for _ in range(epochs):
            for nl, eng in self.trainingdata:
                #self.optimizer.zero_grad()
                for param in self.model.parameters():
                    param.grad = None
                eng_pred = self.model.forward(nl)
                loss = self.lossfn(eng_pred, eng)/nl.shape[0]
                loss.backward()
                self.optimizer.step()

            if self.validationdata == None:
                pass
            else:
                if _ % self.validationinterval == 0 and _ != 0:
                    score = 0
                    self.model = self.model.eval()
                    for nl, eng in self.validationdata:
                        eng_pred = self.model.forward(nl)
                        score += abs((eng-eng_pred).sum())/nl.shape[0]
                    logging.info(score)
                    self.model = self.model.train()
