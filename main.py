from torch import optim
from torch import nn

from configs import configs
from data import train_loader
from loops import train_one_epoch
from model import GPTLanguageModel

if __name__ == '__main__':
    model = GPTLanguageModel(d_model=configs['d_model'],
                             n_block=configs['decoder_layer'],
                             num_heads=configs['num_heads'],
                             dropout=configs['dropout']).to(configs['device'])

    optimizer = optim.SGD(model.parameters(), lr=configs['lr'],  weight_decay=configs['wd'], momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    model, train_loss, = train_one_epoch(model, train_loader, loss_fn, optimizer, 0)
