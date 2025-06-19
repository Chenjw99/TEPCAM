import os
import torch
import argparse
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from model import TEPCAM
from data import TEPDataset, dropInvalid

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit



def train_model(input_file, epoch, learning_rate, d_model, n_heads, save_dir="./ckpts", random_seed=100, GPU_num=0, batch_size=128, model_name='tepcam_test'):
    device = torch.device(f'cuda:{GPU_num}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():print(f'The code runs on GPU{GPU_num}.') 
    df = pd.read_csv(input_file)
    x, y = df[['TCR', 'epitope']], df['Label']
    tcr = x['TCR']
    gss = GroupShuffleSplit(n_splits=1,train_size=.8,random_state=random_seed)

    for i,(train_idx,_) in enumerate(gss.split(x,y,groups=tcr)):
        x_train,y_train = x.iloc[train_idx],y.iloc[train_idx]
        x_train, y_train = dropInvalid(x_train, y_train) # drop invalid sequence
        
    dataset_train = TEPDataset(x_train, y_train, align=True)
    train_loader = DataLoader(dataset_train, batch_size, shuffle=True, drop_last=False)

    model = TEPCAM(d_model=d_model, modelseed=random_seed, n_heads=n_heads)
    model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20],gamma=0.25)

    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    train_loss_list = []
    
    for epoch in range(epoch):
        total_loss = 0
        batches = 0
        model.train()
        for tcrs, peptides, labels, _, _ in tqdm(train_loader):
            tcrs, peptides = tcrs.to(device), peptides.to(device)
            _, _, _, _, _, output = model(tcrs,peptides)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            loss = loss_fn(output, labels) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        train_loss = round(total_loss/batches,4)
        train_loss_list.append(train_loss)
        print(f"Loss for Epoch{epoch + 1}:{train_loss}")
        scheduler.step()
    
    os.makedirs(save_dir,exist_ok = True)
    save_model_file = f'{save_dir}/{model_name}.pt'
    torch.save(model, save_model_file) 

    #Visualization
    os.makedirs(f'./result/pic', exist_ok = True)
    epochs = range(1, len(train_loss_list) + 1)   
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss_list, 'b', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./result/pic/{model_name}.png")

    
def str2bool(s):
	return True if s.upper() == 'TRUE' else False

def create_parser():
    parser = argparse.ArgumentParser(description="Model Training for TEPCAM", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--input_file", dest="input_file", type=str, help="The input file in .csv format.", required=True)
    parser.add_argument("--model_name", dest="model_name", type=str, help="model name and the prefix for saving model.", default="tepcam_test", required=False)
    parser.add_argument("--epoch", dest="epoch", type=int, help="training epoch", default=30, required=False)
    parser.add_argument("--batch_size", dest="batch_size", type=int,help="number of samples per-batch", default=128, required=False)
    parser.add_argument("--d_model", dest="d_model", type=int,help="The dimension of model", default=32, required=False)
    parser.add_argument("--n_heads", dest="n_heads", type=int, help="Number of heads for attention module", default=6, required=False)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.0005, required=False)
    parser.add_argument("--random_seed", dest="random_seed", type=int,default=100, help="Random Seed used in this trial")
    parser.add_argument("--GPU_num", dest="GPU_num", type=int, default=0, help="which GPU to run the code", required=False)
    parser.add_argument("--save_dir", dest="save_dir", type=str, default='./ckpts', help="Folder for saving checkpoint", required=False)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = create_parser() 
    train_model(
        input_file = args.input_file,
        batch_size = args.batch_size,
        epoch = args.epoch,
        learning_rate = args.learning_rate,
        d_model = args.d_model,
        n_heads = args.n_heads,
        random_seed = args.random_seed,
        GPU_num = args.GPU_num,
        model_name = args.model_name,
        save_dir = args.save_dir
        ) 
    print(f"The model training is done! Model saved in {args.save_dir}/{args.model_name}.pt")
    
    
    