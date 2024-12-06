import logging
from torch.utils.tensorboard import SummaryWriter
log_dir = "./log_dir"
from ..model.loss import NTXent
from .utils import *
from tqdm import tqdm
from .evaluation import *


# SimCLR's main learning algorithm
class SimCLR(object):
    # Initialize variables
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()  # tensorboard train log
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training_log'), level=logging.DEBUG)

    def train(self, train_loader):
        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0  # global optimization step
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")

        # train code described as below
        # iterate epoch and train_loader for each epoch
        for epoch_counter in range(self.args.epochs):
            for signals, _ in tqdm(train_loader):
                signals = torch.cat(signals, dim=0)

                # produce image representation <features> (z) from self.model
                # backbone model + MLP outputs
                features = self.model(signals)
                # calculate InfoNCE_loss from features (<-g <-f)
                logits, labels = NTXent.info_nce_loss(features)
                loss = self.criterion(logits, labels)

                # Update network f and g, and minimize loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
