import torch
from neural_network import Neural_Net_CNN

#This class includes functions for saving and loading trained models along with model checkpoints which can be used for future training
class modelLoader:

    def save(filepath, model_state):
        try:
            torch.save(model_state, filepath)
        except IOError:
            print("Error saving Model")

    def load(filepath):
        model = Neural_Net_CNN()
        try:
            model.load_state_dict(torch.load(filepath, weights_only=True))
            model.eval()
        except IOError:
            print("Error loading Model")

    def save_Checkpoint(filepath, epoch, model_state, optimizer_state, loss):
        try:
            torch.save({'epoch': epoch,'model_state_dict': model_state,'optimizer_state_dict': optimizer_state,'loss': loss}, filepath)
        except IOError:
            print("Error saving Model")
            
    #can't ensure a valid optimizer is passed in yet
    #TODO update neural network to be able to pass optimizer into load_checkpoint function
    """def load_Checkpoint(filepath, is_training: bool, optimizer: int):
        model = Neural_Net_CNN()
        checkpoint = torch.load(filepath, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        #one of these needs to be used depending on whether the checkpoint is to b used for further training or for evaluating new data
        if (is_training is True):
            model.train()
        else:
            model.eval()"""

        