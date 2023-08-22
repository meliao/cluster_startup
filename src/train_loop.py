import torch
from torch import nn
import logging
import os

from src.network import add_weight_decay, weight_decay_eval, weight_decay_val
from src.utils import write_result_to_file


def train_one_epoch(dataloader, model, loss_fn, optimizer):
    """
    trains one epoch
    """
    for batch, (X, y) in enumerate(dataloader):
        #Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred.flatten(), y)

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def train_L_layers(model: nn.Module,
                trainX: torch.Tensor,
                trainY: torch.Tensor,
                testX: torch.Tensor,
                testY: torch.Tensor,
                train_results_fp: str,
                models_dir: str,
                weight_decay: float,
                epochs: int,
                n_epochs_per_log: int,
                lr: float=1e-4,
                batch_size: int=64,
                verbose: bool=False,
                no_wd_last_how_many_epochs: int=100,
                scheduler=None,
                device: torch.cuda.Device='cpu',
                **schedulerkwargs) -> None:

    #define pytorch dataloaders
    dataset = torch.utils.data.TensorDataset(trainX,trainY) #create your dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) #create your dataloader

    model.to(device)



    loss_fn = nn.MSELoss()
    paramlist = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.AdamW(paramlist, lr=lr)

    if verbose:
        print("lambda = {}".format(paramlist[1]['weight_decay']))

    #main training loop
    with torch.no_grad():
        trainmse = torch.zeros(epochs,device=device)
        weightdecay = torch.zeros(epochs,device=device)
        learningrate = torch.zeros(epochs,device=device)

    if scheduler is not None:
        scheduler = scheduler(optimizer,**schedulerkwargs)


    apply_weight_decay_flag = True

    for t in range(epochs):
        train_one_epoch(dataloader, model, loss_fn, optimizer)


        # Logging and saving the model weights
        if t % n_epochs_per_log == 0:
            with torch.no_grad():


                #record current MSE, weight decay value, and learning rate
                train_mse = loss_fn(model(trainX).flatten(), trainY).item()
                weight_decay_t = weight_decay_eval(paramlist).item()
                test_mse = loss_fn(model(testX).flatten(), testY).item()

                if scheduler is not None:
                    learningrate = scheduler.optimizer.param_groups[0]['lr']
                else:
                    learningrate = None

                logging.info("Epoch %i/%i. Train MSE: %f, Test MSE: %f, Weight decay: %f",
                            t,
                            epochs,
                            train_mse,
                            test_mse,
                            weight_decay_t)
                
                # Save the results to a text file
                out_dd = {
                    'epoch': t,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'weight_decay': weight_decay_t,
                    'lr': learningrate,
                }
                write_result_to_file(train_results_fp, **out_dd)
            
                # Save the model
                model_fp = os.path.join(models_dir, f'epoch_{t}.pickle')
                torch.save(model.state_dict(), model_fp)


        #adjust learning rate
        if scheduler is not None:
            scheduler.step()

        #turn off weight decay for last 100 epochs
        if apply_weight_decay_flag is False and t > epochs - no_wd_last_how_many_epochs:
            optimizer.param_groups[1]['weight_decay'] = 0
            apply_weight_decay_flag = True
