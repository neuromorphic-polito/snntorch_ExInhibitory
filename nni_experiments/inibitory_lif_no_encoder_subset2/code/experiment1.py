"""Python script for training with nni. This file is not called with python.
Instead we use nnictl and the yml configuration file."""

import argparse
import os, sys
from pathlib import Path
import nni.tools
import nni.tools.nnictl
import nni.tools.nnictl.updater
from tqdm import tqdm

from  config import *
import nni
from torch.utils.data import  DataLoader

from snntorch import surrogate
from snntorch import functional as SF


from snntorch_network.dataloader import *
from snntorch_network.utils import *
from snntorch_network.my_network import *
from snntorch_network.assistant import Assistant
from snntorch_network.stats import LearningStats


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_path', type=str, help='nome del config file per creare la cartella adeguata')
    args = parser.parse_args()

    # nni will get the parameters from the file that has been indicated
    # in searchSpaceFile in the config.yml file. In this case,
    # search_space1.json

    params = nni.get_next_parameter()

    ### Every n_tr trials, "update" the searchspace inducing a new RandomState for the tuner
    n_tr = SEARCH_SPACE_SHUFFLE
    searchspace_path = f'{Path.home()}/snntorch_network/nni_experiments/{args.trial_path}/search_space/search_space1.json'
    updated_searchspace = SearchSpaceUpdater({"filename": searchspace_path, "id": nni.get_experiment_id()})
    if (nni.get_sequence_id() > 0) & (nni.get_sequence_id()%n_tr == 0):
        nni.tools.nnictl.updater.update_searchspace(updated_searchspace) # it will use update_searchspace.filename to update the search space
        print(f'Updated searchspace at trial {nni.get_sequence_id()}')

    os.chdir(f'{Path.home()}/snntorch_network/nni_experiments/{args.trial_path}/results/{nni.get_experiment_id()}/environments/local-env/trials/{nni.get_trial_id()}')
    trained_folder = TRAIN_FOLDER_NAME
    os.makedirs(trained_folder, exist_ok=True)

    dataset = WisdmDatasetParser(f'{Path.home()}/snntorch_network/data/{DATASET_NAME}', norm=None, class_sublset=DATASET_SUBSET, subset_list=SUBSET_LIST)
    train_set = dataset.get_training_set()
    val_set = dataset.get_validation_set()

    data, label = train_set
    print(data.shape)

    train_dataset = WisdmDataset(train_set)
    val_dataset = WisdmDataset(val_set)

    train_loader = DataLoader(dataset=train_dataset, batch_size=int(params['batch_size']), shuffle=True, num_workers=NUM_WORKERS)
    val_loader  = DataLoader(dataset= val_dataset, batch_size=int(params['batch_size']), shuffle=True, num_workers=NUM_WORKERS)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f'Using device {device}')

    grad = surrogate.fast_sigmoid(params['slope']) #use slope for HPO

    net_loss = regularization_loss(0.1, 0.03, 40, device=device)

    net = ExInhbitoryNetwork(NET_INPUT_DIM, int(params['net_hidden_1']), int(params['net_hidden_2']), NET_OUTPUT_DIM, grad,
                        vth_in=params['vth_in'], vth_recurrent=params['vth_recurrent'], vth_out=params['vth_out'], vth_back=params['vth_back'],
                        beta_in=params['beta_in'], beta_recurrent=params['beta_recurrent'], beta_back=params['beta_back'], beta_out=params['beta_out'],
                        drop_recurrent=params['drop_recurrent'], drop_back=params['drop_back'], drop_out=params['drop_out'], time_dim=2, layer_loss=net_loss).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'], betas=(0.9, 0.999))

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=4690,
        eta_min=0,
        last_epoch=-1
    )

    # Currently only using cross-entropy
    if params['loss_fn'] == 'mse_count_loss':
        loss_fn = SF.loss.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2) #param loss for HPO
    elif params['loss_fn'] == 'ce_count_loss':
        loss_fn = SF.loss.ce_count_loss()

    # The stuff to make things easier (inspired by lava-dl)
    stats = LearningStats()
    assistant = Assistant(net, loss_fn, optimizer, stats, classifier=True, scheduler=scheduler, lam=1.0)

    count = 0
    for epoch in range(NUM_EPOCHS):
        labels = []
        outputs = []
        # if epoch % 20 == 0:
        #     assistant.reduce_lr()


        # if for PATINCE epochs the accuracy does not improve, finish the
        # training. (The remaining epochs will not do anything)
        if count < PATIENCE:
            count = count+1
            tqdm_dataloader = tqdm(train_loader)
            for _, batch in enumerate(tqdm_dataloader): # training loop
                input, label = batch

                assistant.train(input, label)
                tqdm_dataloader.set_description(f'\r[Epoch {epoch:2d}/{NUM_EPOCHS}] Training: {stats.training}')

            tqdm_dataloader = tqdm(val_loader)
            for _, batch in enumerate(tqdm_dataloader): #eval loop
                input, label = batch
                output = assistant.valid(input, label)
                tqdm_dataloader.set_description(f'\r[Epoch {epoch:2d}/{NUM_EPOCHS}] Validation: {stats.validation}')

                # Concatenate all validation outputs
                if len(outputs) == 0:
                    outputs = output.to('cpu').detach()
                    labels = label.to('cpu').detach()
                else:
                    outputs = torch.cat((outputs, output.to('cpu').detach()), dim=1)
                    labels = torch.cat((labels, label.to('cpu').detach()))

            # Log intermediate results to nni, for logging, plots, etc.
            nni.report_intermediate_result(stats.validation.accuracy*100)

            stats.update()

            # this is true if the current is the best
            if stats.validation.best_accuracy:
                count = 0

                # We get the prediction as the class that has produced more
                # spikes during the simulation. So we sum all the spikes, and
                # get the class with the max sum.
                _, predictions = outputs.sum(dim=0).max(1)
                gen_confusion_matrix(predictions,labels, f'./{trained_folder}/')

                # Save the currently best network.
                net.save_to_npz(f'./{trained_folder}/network_best.npz')
                del predictions

            del outputs
            del labels

            torch.cuda.empty_cache()

    # Plot of loss and accuracy of all the epochs, for training and val
    stats.plot(figsize=(15, 5),path=f'./{trained_folder}/')

    # Save .txt with loss and accuracy of all the epochs, for training and val
    stats.save( f'./{trained_folder}/')

    # Report the final result, which is the with the best accuracy
    # So nni can understand the results of this configuration, and use it
    # to find the next one to try.
    nni.report_final_result(stats.validation.max_accuracy*100)


if __name__ == '__main__':
    main()
