import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torcheval.metrics import Mean
from tqdm import tqdm
from configs import configs


def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch=None):

    model.train()
    loss_train = Mean().to(configs['device'])
    with tqdm(train_loader, unit='batch') as tepochs:
        for x_batch, y_batch in tepochs:
            if epoch is not None:
                tepochs.set_description(f'epoch:{epoch}')
            x_batch = x_batch.to(configs['device'])
            y_batch = y_batch.to(configs['device'])
            yp = model(x_batch.to(configs['device']))

            loss = loss_fn(yp.transpose(2, 1), y_batch)

            loss.backward()
            clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            optimizer.zero_grad()
            tepochs.set_postfix(loss=loss_train.compute().item())
            loss_train.update(loss)
        loss = loss_train.compute()
    return model, loss


def evaluate(model, test_loader, loss_fn):
    model.eval()
    loss_test = Mean().to(configs['device'])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(configs['device'])
            y_batch = y_batch.to(configs['device'])
            yp = model(x_batch)
            loss = loss_fn(yp.transpose(2, 1), y_batch)
            loss_test.update(loss)
    print(f'loss : {loss_test.compute()}')
    return loss_test.compute().item()
