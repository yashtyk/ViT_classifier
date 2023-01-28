import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from ViT import MyViT



from dataloader_casa import CASA_dataset

import wandb



#training

def main():
    # creating dataset
    dataset = CASA_dataset()

    # division by train, validation, test

    train_per = 0.8
    val_per = 0.1
    test_per = 0.1

    train_size = int(len(dataset) * train_per)
    val_size = int(len(dataset) * val_per)
    test_size = len(dataset) - train_size - val_size

    # fixed seed
    seed = 0

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(seed))



    # creating dataloader

    batch_size = 128

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)


    # Defining model and training options
    n_channel  = 1
    picture_size = 512
    n_patches = 8
    n_blocks = 2
    hidden_d = 1024
    n_heads = 8
    out_d = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = MyViT((n_channel, picture_size, picture_size), n_patches = n_patches, n_blocks = n_blocks, hidden_d = hidden_d, n_heads = n_heads, out_d=out_d, device= device).to(device)

    print('model device: ')
    print(next(model.parameters()).device)

    n_epochs = 100
    limit_wait_epochs = 20
    lr = 0.005

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    #training loop
    wait_epochs = 0
    etalon_loss = 100000
    best_epochs = -1

    print('training...')
    # connect to the weights and biases

    wandb.init(project = "ViT", entity = "yshtyk")

    wandb.config= {
        "learning_rate": 0.001,
        "batch_size": 128

    }

    for epoch in range(n_epochs):
        print('training epoch: {}'.format(epoch))
        train_loss= 0.0
        for batch in train_dataloader:
            print('new batch size')
            x, y1 = batch
            #y1 =  y1.to(torch.float32) - 1
            y1= y1 - 1
            x, y1 = x.to(device), y1.to(device)
            print('device of x: ')
            print(x.device)

            y_hat = model(x)
            y = y1.squeeze()
            loss= criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().item() / len(train_dataloader)






        # validation
        with torch.no_grad():
            print('validation..')
            val_loss = 0.0
            for batch in val_dataloader:
                print('new batch size')
                x, y1 = batch
                #y1 = y1.to(torch.float32)
                y1 = y1 - 1
                x, y1  = x.to(device), y1.to(device)
                y_hat = model(x)
                y = y1.squeeze()
                loss = criterion(y_hat, y)
                val_loss += loss.detach().cpu().item() / len(val_dataloader)

            print('epoch: {}, train_loss: {}, val_loss: {}'.format(epoch, train_loss, val_loss))



            if val_loss < etalon_loss:
                wait_epochs = 0
                etalon_loss = val_loss
                best_epochs = epoch
                torch.save(model.state_dict(), '/srv/beegfs/scratch/users/s/shtyk1/vit_casa/classifier/model.pt')
                print('new etalon saved, epoch: {}, val_loss: {:.4f}'.format(epoch, val_loss))

            elif wait_epochs <= limit_wait_epochs:
                wait_epochs +=1

            else:
                print('limit of the wait epochs reached, best epoch: {}, loss:{} '.format(best_epochs, etalon_loss))
                break

        # login loss to the weights and biases
        wandb.log({"train_loss": train_loss,
                   "val_loss": val_loss,
                   "waiting_epochs": wait_epochs,
                   "best_epochs": best_epochs})


    print('training ended, best epoch: {}, val_loss: {}'.format(epoch, val_loss))

    print('testing...')



    with torch.no_grad():

        # load model weights
        model.load_state_dict(torch.load('/srv/beegfs/scratch/users/s/shtyk1/vit_casa/classifier/models.pt'))
        test_loss =0.0
        for batch in test_dataloader:
            x, y1 = batch
            #y1 = y1.to(torch.float32)
            y1= y1 - 1
            x, y1 = x.to(device), y1.to(device)
            y_hat = model(x)
            y = y1.squeeze()
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_dataloader)

    print('test_loss: {}'.format(test_loss))

if __name__=='__main__':
    main()


