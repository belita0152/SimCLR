# Load libraries and files
import torch
from dataset.parser import Dataset
from dataset.augmentation import SignalAugmentation

# Define Args

# main
if __name__ == "__main__":
    # parser 지정

    # Load data
    dataset = Dataset()
    data_list, info_list = dataset.parser()

    # Data Augmentation
    augmentation = SignalAugmentation()




    # Load train, valid dataset
    train_dataset = data_list[:5]  # 일단 전체로 돌리기
    # valid_dataset = data_list[4:]


    # Put dataset in torch DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, drop_last=True
    )


    '''
    # load dataset while setting data path
    dataset = ContrastiveLearningDataset(args.data)

    # load valid dataset with defined transform
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    # put dataset in torch DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True
    )

    # load base encoder f (EEGNet)
    model = EEGNet(base_model=args.arch, out_dim=args.out_dim)

    # set optimizer
    base_optimizer = optim.___(model.parameters(), lr=0.1)  # SGD or else
    optimizer = ____(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)  # LARS or else
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=len(train_loader),
                                                           eta_min=0,
                                                           last_epoch=-1)

    # initialize SimCLR model and train it
    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train(train_loader)
    '''



