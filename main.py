import os
import argparse
import json
import pandas as pd
import math
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models import Generalized_Matrix_Factorization, Neural_Collaborative_Filtering, Neural_Matrix_Factorization
import utils
import dataset

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument("--epoch", type=int, default=100, help="training epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for mini-batch train")

    parser.add_argument("--embed_dim", type=int, default=32, help="embedding size of MF part")
    parser.add_argument("--layers", nargs="+", type=int, default=[64, 32, 16, 8],
                        help="MLP layers, first layer is CONCAT of user/item embeddings (layer[0]/2 is input embed size of MLP part)")
    
    parser.add_argument("--name", type=str, required=True, help="model name for logging & saving")
    parser.add_argument("--model_type", type=str, required=True, help="GMF or NCF or NeuMF")

    args = parser.parse_args()

    return args

def train(args, model, device, train_loader, valid_loader, optimizer, loss_function, epoch, best_rmse, best_mae, checkpoint_path, writer):
    """Train & Valid (every epoch)"""
    train_losses = AverageMeter()

    epoch_iterator_train = tqdm(train_loader, desc="Training (X / X Steps) (loss = X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)

    # Train step
    model.train()
    for step, list_data in enumerate(epoch_iterator_train):
        user = list_data[0].to(device)
        item = list_data[1].to(device)
        rating = list_data[2].to(device)

        optimizer.zero_grad()

        prediction = model(user, item)

        loss = loss_function(prediction, rating)

        loss.backward()
        optimizer.step()

        train_losses.update(loss)

        epoch_iterator_train.set_description("Training (%d / %d Steps) (loss = %2.5f)" % (step, len(epoch_iterator_train), train_losses.val))
    
    # Valid step
    eval_losses = AverageMeter()
    tmp_score, target_score = [], []
    
    model.eval()
    with torch.no_grad():
        epoch_iterator_valid = tqdm(valid_loader, desc="Validating (X / X Steps) (loss = X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)

        for step, list_data in enumerate(epoch_iterator_valid):
            user = list_data[0].to(device)
            item = list_data[1].to(device)
            rating = list_data[2].to(device)

            prediction = model(user, item)

            loss = loss_function(prediction, rating)

            eval_losses.update(loss)

            tmp_score.append(list(prediction.data.cpu().numpy()))
            target_score.append(list(rating.data.cpu().numpy()))

            epoch_iterator_valid.set_description("Validating (%d / %d Steps) (loss = %2.5f)" % (step, len(epoch_iterator_valid), eval_losses.val))
        
        tmp_score = np.array(sum(tmp_score, []))
        target_score = np.array(sum(target_score, []))

        total_rmse = math.sqrt(mean_squared_error(tmp_score, target_score))
        total_mae = mean_absolute_error(tmp_score, target_score)

        if total_rmse < best_rmse:
            best_rmse, best_mae = total_rmse, total_mae
            torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
            print(f"\t\t best model saved: epoch = {epoch}, valid RMSE = {total_rmse:.6f}, valid MAE = {total_mae:.6f}")
    
    # Tensorboard logging
    writer.add_scalars('Loss', {'Train':train_losses.avg, 'Valid':eval_losses.avg}, epoch)
    writer.add_scalar('RMSE/Valid', total_rmse, epoch)
    writer.add_scalar('MAE/Valid', total_mae, epoch)
    
    print(f"Epoch {epoch:03d}: Train Loss: {train_losses.avg:.4f} || Valid Loss: {eval_losses.avg:.4f} || epoch RMSE: {total_rmse:.4f} || epoch MAE: {total_mae:.4f} || best RMSE: {best_rmse:.4f} || best MAE: {best_mae:.4f}")
    return best_rmse, best_mae


def test(args, model, device, test_loader, loss_function, checkpoint_path):
    # load best state model (best valid model)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading the best model from: " + checkpoint_path)

    eval_rmse = []
    eval_mae = []
    test_losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        epoch_iterator_test = tqdm(test_loader, desc="Evaluating (X / X Steps) (loss = X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)

        for step, list_data in enumerate(epoch_iterator_test):
            user = list_data[0].to(device)  
            item = list_data[1].to(device)
            rating = list_data[2].to(device)

            prediction = model(user, item)

            loss = loss_function(prediction, rating)

            test_losses.update(loss)

            mse = F.mse_loss(prediction, rating, reduction='none')
            rmse = torch.sqrt(mse.mean())
            mae = F.l1_loss(prediction, rating, reduction='mean')

            eval_rmse.append(rmse)
            eval_mae.append(mae)

            epoch_iterator_test.set_description("Evaluating (%d / %d Steps) (loss = %2.5f)" % (step, len(epoch_iterator_test), test_losses.val))
        
        total_rmse = sum(eval_rmse) / len(eval_rmse)
        total_mae = sum(eval_mae) / len(eval_mae)
    
    print("\n [Evaluation Results]")
    print("Loss: %2.5f" % test_losses.avg)
    print("RMSE: %2.5f" % total_rmse)
    print("MAE: %2.5f" % total_mae)

    return total_rmse, total_mae



def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set random seed
    utils.seed_everything(args.seed)

    # configure model checkpoint dir
    checkpoint_dir = os.getcwd() + '/checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_dir = checkpoint_dir + f'checkpoints_seed_{args.seed}/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_path = checkpoint_dir + f'{args.name}.{args.model_type}.model'

    # configure logging dir
    log_dir = os.getcwd() + '/logs/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = log_dir + f'log_seed_{args.seed}'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    # load data
    print("Loading data...")
    data_path = os.getcwd() + '/KuaiRec/data/'
    train_df = pd.read_csv(data_path + 'big_matrix.csv')
    test_df = pd.read_csv(data_path + 'small_matrix.csv')

    # index starts from 0
    num_users = train_df['user_id'].nunique()   # 7,176 users
    num_items = train_df['video_id'].nunique()  # 10,728 items

    # prepare data
    train_data = dataset.MyDataset(args, train_df, 'big')
    test_data = dataset.MyDataset(args, test_df, 'small')

    train_loader = train_data.load_train_data()
    valid_loader = train_data.load_valid_data()
    test_loader = test_data.load_test_data()

    print("Data loading finished...")

    ## batch size 256, train:44054, valid:4895, test:18268
    # print(len(train_loader), len(valid_loader), len(test_loader))

    # prepare training
    if args.model_type == 'GMF':
        model = Generalized_Matrix_Factorization(args, num_users, num_items)
    elif args.model_type == 'NCF':
        model = Neural_Collaborative_Filtering(args, num_users, num_items)
    elif args.model_type == 'NeuMF':
        model = Neural_Matrix_Factorization(args, num_users, num_items)
    else:
        raise Exception("Unknown model type")

    model = model.to(device)
    
    loss_function = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

    writer = SummaryWriter(os.path.join(log_dir, f"{args.name}.{args.model_type}.tensorboard"))

    valid_RMSE, valid_MAE = 9999.0, 9999.0

    start_time = time.time()
    for epoch in range(1, args.epoch + 1):
        valid_RMSE, valid_MAE = train(args, 
                                    model, 
                                    device, 
                                    train_loader, 
                                    valid_loader, 
                                    optimizer, 
                                    loss_function,
                                    epoch,
                                    valid_RMSE,
                                    valid_MAE,
                                    checkpoint_path,
                                    writer)
    end_time = time.time()
    
    # logging start
    log_path = os.path.join(log_dir, f'{args.name}.{args.model_type}.log')
    utils.redirect_stdout(open(log_path, 'w'))

    print(json.dumps(args.__dict__, indent=4))

    print('\n')
    print(model)
    print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

    print('\n')
    print("########### Result ###########")
    print(f"Total train time: {end_time - start_time:.4f}s")

    test_RMSE, test_MAE = test(args,
                               model,
                               device,
                               test_loader,
                               loss_function,
                               checkpoint_path)

    print('\n')
    print(f"Test RMSE: {test_RMSE:.4f} (Best valid RMSE: {valid_RMSE:.4f})")
    print(f"Test MAE: {test_MAE:.4f} (Best valid MAE: {valid_MAE:.4f})")
    

if __name__ == "__main__":
    main()