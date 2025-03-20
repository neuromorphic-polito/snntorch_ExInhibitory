import argparse
import os, sys
from pathlib import Path
from tqdm import tqdm

from  config import *
import nni
from torch.utils.data import  DataLoader

from snntorch import surrogate
from snntorch import functional as SF

sys.path.insert(0, '../../../src/')
from dataloader import *
from utils import *
from networks_old import *
from assistant import Assistant
from stats import LearningStats

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_path', type=str, help='nome del config file per creare la cartella adeguata')
    args = parser.parse_args()

    params = nni.get_next_parameter()

    os.chdir(f'{Path.home()}/snntorch_network/nni_experiments/{args.trial_path}/{nni.get_experiment_id()}/trials/{nni.get_trial_id()}')
    trained_folder = TRAIN_FOLDER_NAME
    os.makedirs(trained_folder, exist_ok=True)

    dataset = WisdmEncodedDatasetParser(f'{Path.home()}/snntorch_network/data/{DATASET_NAME}')
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

    net = ExInbitoryNetwork(NET_INPUT_DIM, int(params['net_hidden_1']), int(params['net_hidden_2']),
                      NET_OUTPUT_DIM, params['beta'],grad, params['threshold'], back_beta=params['back_beta']).to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
        T_max=4690, 
        eta_min=0, 
        last_epoch=-1
    )
    if params['loss_fn'] == 'mse_count_loss':
        loss_fn = SF.loss.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2) #param loss for HPO
    elif params['loss_fn'] == 'ce_count_loss':
        loss_fn = SF.loss.ce_count_loss()

    stats = LearningStats()
    assistant = Assistant(net, loss_fn, optimizer, stats, classifier=True, scheduler=scheduler)

    count = 0
    for epoch in range(NUM_EPOCHS):
        labels = []
        outputs = []
        # if epoch % 20 == 0:
        #     assistant.reduce_lr()
        if count < PATIENCE:
            count = count+1
            tqdm_dataloader = tqdm(train_loader)
            for _, batch in enumerate(tqdm_dataloader): # training loop
                input, label = batch
                
                output = assistant.train(input, label)
                tqdm_dataloader.set_description(f'\r[Epoch {epoch:2d}/{NUM_EPOCHS}] Training: {stats.training}')

            tqdm_dataloader = tqdm(val_loader)
            for _, batch in enumerate(tqdm_dataloader): #eval loop
                input, label = batch
                output = assistant.test(input, label)
                tqdm_dataloader.set_description(f'\r[Epoch {epoch:2d}/{NUM_EPOCHS}] Validation: {stats.testing}')
            
                if len(outputs) == 0:
                    outputs = output.to('cpu').detach()
                    labels = label.to('cpu').detach()
                else:
                    outputs = torch.cat((outputs, output.to('cpu').detach()), dim=1)
                    labels = torch.cat((labels, label.to('cpu').detach()))

            nni.report_intermediate_result(stats.testing.accuracy*100)

            stats.update()

            if stats.testing.best_accuracy:
                count = 0
                _, predictions = outputs.sum(dim=0).max(1)
                gen_confusion_matrix(predictions,labels, f'./{trained_folder}/')
                net.save_to_npz(f'./{trained_folder}/network_best.npz')
                stats.save( f'./{trained_folder}/')

            del outputs
            del labels
            torch.cuda.empty_cache()
        
    nni.report_final_result(stats.testing.max_accuracy*100)
    stats.plot(figsize=(15, 5),path=f'./{trained_folder}/')

if __name__ == '__main__':
    main()
