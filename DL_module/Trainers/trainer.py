"""
,---------. .-------.       ____    .-./`) ,---.   .--.    .-''-.  .-------.
\          \|  _ _   \    .'  __ `. \ .-.')|    \  |  |  .'_ _   \ |  _ _   \
 `--.  ,---'| ( ' )  |   /   '  \  \/ `-' \|  ,  \ |  | / ( ` )   '| ( ' )  |
    |   \   |(_ o _) /   |___|  /  | `-'`"`|  |\_ \|  |. (_ o _)  ||(_ o _) /
    :_ _:   | (_,_).' __    _.-`   | .---. |  _( )_\  ||  (_,_)___|| (_,_).' __
    (_I_)   |  |\ \  |  |.'   _    | |   | | (_ o _)  |'  \   .---.|  |\ \  |  |
   (_(=)_)  |  | \ `'   /|  _( )_  | |   | |  (_,_)\  | \  `-'    /|  | \ `'   /
    (_I_)   |  |  \    / \ (_ o _) / |   | |  |    |  |  \       / |  |  \    /     _
    '---'   ''-'   `'-'   '.(_,_).'  '---' '--'    '--'   `'-..-'  ''-'   `'-     _( )_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(_ o _)  Models' Trainer
                                                                                  (_,_)
"""

import dgl
import torch
import torch.nn.functional as F
from abc import abstractmethod
from numpy import inf
import numpy as np
from time import time

from torchvision.utils import make_grid

from Logger import TensorboardWriter
from Utils.misc import seconds2hms

from .trainer_util import *


# ================================┌------------------┐================================
#                                 | pytorch-template |
# ================================└------------------┘================================

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        save_untrained:bool=True,
    ):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:#if config.resume is not None:
            self._resume_checkpoint(config.resume)
            
        self.save_untrained = save_untrained

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        if self.save_untrained:
            self._save_checkpoint(0, save_name="model_untrained.pth", save_best_only=True, save_best=False)
        
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_time = time()
            result = self._train_epoch(epoch)
            epoch_time = time() - epoch_time
            result["time"] = seconds2hms(epoch_time)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                log["no improvement"] = not_improved_count

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            if epoch % self.save_period == 0 or best:
                best_only = not epoch % self.save_period == 0
                self._save_checkpoint(epoch, save_best=best, save_best_only=best_only)

    def _save_checkpoint(self, epoch, save_best=False, save_best_only=False, save_name:str=""):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if len(save_name) and save_name.endswith('.pth'): 
            given_path = str(self.checkpoint_dir / save_name)
            torch.save(state, given_path)
            self.logger.info("Saving checkpoint: {} ...".format(given_path))
            
        if not save_best_only:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Model previously trained for {} epochs.".format(self.start_epoch-1))
        

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        metric_tracker_keys = ['loss']
        if self.metric_ftns is not None:
            metric_tracker_keys += [m.__name__ for m in self.metric_ftns]
        self.train_metrics = MetricTracker(*metric_tracker_keys, writer=self.writer)
        self.valid_metrics = MetricTracker(*metric_tracker_keys, writer=self.writer)

    @abstractmethod
    def _batch_loss(self, idx, batch):
        """
        """
        raise NotImplementedError

    def _batch_step(self, idx, batch, epoch):
        self.optimizer.step()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            loss = self._batch_loss(batch_idx, batch)
            loss.backward()

            self._batch_step(batch_idx, batch, epoch)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            if self.metric_ftns is not None:
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)



# =================================X----------------X=================================
#                                   GraphMAE Trainer
# =================================X----------------X=================================

class GraphMAETrainer(Trainer):
    """
    TODOs:
    - [x] modify GraphMAEModel.forward to output reconstructed graph and use criterion here

    """
    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None
    ):
        model.input_dimension = data_loader.dataset.feature_dim
        super().__init__(
            model,
            criterion,
            metric_ftns,
            optimizer,
            config,
            device,
            data_loader,
            valid_data_loader,
            lr_scheduler,
            len_epoch
        )

    def _batch_loss(self
                    , idx
                    , batch
                   ):
        if self.data_loader.dataset._has_labels:
            batch_g, _ = batch
        else:
            batch_g = batch
        batch_g = batch_g.to(self.device)

        feat = batch_g.ndata["attr"]

        x_rec, x_init = self.model(batch_g, feat)
        #loss, loss_dict = self.model(batch_g, feat)
        loss = self.criterion(x_rec, x_init)

        return loss

    # ====================================================
    # ⚠️ Deprecated versions

    def _batch_loss_deprecated(self
                    , idx
                    , batch
                   ):
        if self.data_loader.dataset._has_labels:
            batch_g, _ = batch
        else:
            batch_g = batch
        batch_g = batch_g.to(self.device)

        feat = batch_g.ndata["attr"]
        loss, loss_dict = self.model(batch_g, feat)

        return loss


# =================================X----------------X=================================
#                                     GCC Trainer
# =================================X----------------X=================================

class GCCTrainer(Trainer):
    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,

        model_ema=None,
        finetune: bool = False,
        moco: bool = True,
        ft: bool = False,
        alpha_moco: float = 0.0002,
        clip_norm: float = 9999,
        warmup: float = 0.1,
        lr : float = 0.001,

        contrast_nce_k:int=32,
        contrast_nce_t:float=0.07,
        contrast_softmax:bool=True,
    ):
        super().__init__(
            model,
            criterion,
            metric_ftns,
            optimizer,
            config,
            device,
            data_loader,
            valid_data_loader,
            lr_scheduler,
            len_epoch
        )

        from copy import deepcopy

        self._finetune = finetune
        self._moco = moco
        self._ft = ft
        self._alpha = alpha_moco
        self.model_ema = model_ema
        if self._moco and (self.model_ema is None):
            self.model_ema = deepcopy(self.model)
            #raise ValueError("'moco' requires 'model_ema'.")
        if self._moco:
            # copy weights from `model' to `model_ema'
            self._moment_update(0) #useless ?

        self._clip_norm = clip_norm
        self._warmup = warmup
        self._lr = lr
        self._nce_t = contrast_nce_t

        self.contrast = MemoryMoCo(
            #inputSize=self.model.encoder.hidden_dimension
            inputSize=self.model.rdt_output_dimension
            , outputSize=None
            , K=contrast_nce_k
            , T=contrast_nce_t
            , use_softmax=contrast_softmax
            , todevice = device
        )

    def _moment_update(self, m=None):
        if m is None:
            m = self._alpha
        """ model_ema = m * model_ema + (1 - m) model """
        for p1, p2 in zip(self.model.parameters(), self.model_ema.parameters()):
            p2.data.mul_(m).add_(1 - m, p1.detach().data)

    def _batch_loss(self
                    , idx
                    , batch
                   ):
        if self._moco or not self._ft:
            return self.loss_moco(batch)
        else:
            return self.loss_ft(batch)

    def loss_moco(self
                  , batch
                 ):
        graph_q, graph_k = batch
        graph_q, graph_k = graph_q.to(self.device), graph_k.to(self.device)
        
        bsz = graph_q.batch_size
            
        if self._moco:
            self.model_ema.eval()
            def set_bn_train(m):
                classname = m.__class__.__name__
                if classname.find("BatchNorm") != -1:
                    m.train()

            self.model_ema.apply(set_bn_train)

            # ===================Moco forward=====================
            feat_q = self.model(graph_q)
            with torch.no_grad():
                feat_k = self.model_ema(graph_k)

            out = self.contrast(feat_q, feat_k)
            prob = out[:, 0].mean()
            
        else:
            # ===================Negative sampling forward=====================
            feat_q = self.model(graph_q)
            feat_k = self.model(graph_k)

            out = torch.matmul(feat_k, feat_q.t()) / self._nce_t#opt.nce_t
            prob = out[range(graph_q.batch_size), range(graph_q.batch_size)].mean()

        #assert feat_q.shape == (graph_q.batch_size, model.hidden_dimension)

        loss = self.criterion(out)

        return loss

    def loss_ft(self
                , batch
               ):
        raise NotImplementedError("Fine-tune loss not implemented yet for GCC. \n(adapt code from https://github.com/THUDM/GCC/blob/20398aac95957784865d6c78bc46ead605221f0d/train.py#L175 if necessary.")


    def _batch_step(self, idx, batch, epoch):
        #grad_norm = clip_grad_norm(self.model.parameters(), self._clip_norm)

        n_batch = len(self.data_loader)
        global_step = epoch * n_batch + idx
        lr_this_step = self._lr * warmup_linear(
            global_step / (self.epochs * n_batch), self._warmup
        )
        lr_this_step = self._lr # temporary: constant LR
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_this_step

        self.optimizer.step()

        if self._moco:
            self._moment_update()

        #torch.cuda.synchornize()

        
# =================================X--------------X===================================
#                                    PGCL Trainer                                    
# =================================X--------------X===================================


class PGCLTrainer(Trainer):
    """
    TODOs:
    - [x] modify GraphMAEModel.forward to output reconstructed graph and use criterion here

    """
    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
        
        # PGCL
        use_queue:bool=True,
        crops_for_assign:list=[0,1],
        nmb_crops:list=[2],
        queue_length:int=512,
        
        #temperature:float=0.2,
        #hard_selection:bool=True,
        #sample_reweighting:bool=False,
        
        #epsilon:float=0.05,
        #world_size:int=1,
        #sinkhorn_iterations:int=3,
    ):
        #model.input_dimension = data_loader.dataset.feature_dim
        super().__init__(
            model,
            criterion,
            metric_ftns,
            optimizer,
            config,
            device,
            data_loader,
            valid_data_loader,
            lr_scheduler,
            len_epoch
        )
        
        self.use_queue = use_queue
        self.queue_length = queue_length
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops
        
        if self.use_queue:
            self.init_queue()
        else:
            self.queue = None
        
        #self.temperature = temperature
        #self.hard_selection = hard_selection
        #self.sample_reweighting = sample_reweighting
        
        #self.sinkhorn_iterations = sinkhorn_iterations
        #self.epsilon = epsilon
        #self.world_size = world_size
        
        
        self.global_emb, self.global_output, self.global_prot, self.global_y, self.global_plabel = [], [], [], [], []
        
    
    def init_queue(self, inplace=True):
        queue = torch.zeros(
            len(self.crops_for_assign),
            self.queue_length,
            self.model.encoder.output_dimension,# * self.model.encoder.num_gc_layers,
        ).to(self.device)
        
        if inplace:
            self.queue = queue
            return
        else:
            return queue
        
    """
    def loss_cal(self
                 , x
                 , x_aug
                 , hard_q
                 , neg_q
                ):
        
        batch_size = x.size(0)

        out_1 = F.normalize(x, dim=1)
        out_2 = F.normalize(x_aug, dim=1)

        # neg score  [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        neg_q = torch.cat([neg_q[0], neg_q[1]], dim=0)
        hard_q = torch.cat([hard_q[0], hard_q[1]], dim=0)
        # neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        sim_matrix  = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()

        # compute distances among prototypes
        w = self.model.prototypes.weight.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        proto_dist = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)

        # proto_dist = F.normalize(proto_dist, dim=1)

        # negative samples selection
        if self.hard_selection:
            for i, row in enumerate(mask):
                for j, col in enumerate(row):
                    if hard_q[j] not in neg_q[i]:
                        mask[i][j] = False

        if self.sample_reweighting:
            reweight = torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)
            for i, row in enumerate(sim_matrix):
                for j, col in enumerate(row):
                    if i != j:
                        # obtain the prototype id
                        q_i, q_j = hard_q[i].item(), hard_q[j].item()
                        reweight[i][j] = proto_dist[q_i][q_j]
                reweight[i][i] = torch.min(reweight[i])
                # MaxMin scaler
                r_min, r_max = torch.min(reweight[i]), torch.max(reweight[i])
                reweight[i] = (reweight[i] - r_min) / (r_max - r_min)
            # print("before:{}".format(reweight))

            mu = torch.mean(reweight, dim=1)
            std = torch.std(reweight, dim=1)
            # reweight = (reweight - mu) / std
            reweight = torch.exp(torch.pow((reweight - mu), 2) / (2 * torch.pow(std, 2)))
            # print("after:{}".format(reweight))
            sim_matrix = sim_matrix * reweight

        sim_matrix  = sim_matrix.masked_select(mask)#.view(2 * batch_size, -1)
        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos = torch.cat([pos, pos], dim=0)

        loss = (- torch.log(pos / sim_matrix.sum(dim=-1))).mean()


        return loss
    """
        
    def _batch_loss(self
                    , idx
                    , batch
                   ):
        # re-initialise queue on first batch
        if idx==0: 
            self.init_queue()
        
        data, data_aug, data_stro_aug = batch

        node_num = data.num_nodes()
        data_dl = data.to(self.device)

        bs = len(dgl.unbatch(data))

        if idx == 0:
            self.global_prot.append(self.model.prototypes.weight)

        # normalize the prototypes
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            # print("w:{}".format(w))
            self.model.prototypes.weight.copy_(w)


        # ============ forward passes ... ============
        # feature, scores
        #embedding, output = self.model(data.x, data.edge_index, data.batch, data.num_graphs)
        embedding, output = self.model(data)
        #print(output.shape)
        #print(embedding.shape)
        
        self.global_emb.append(embedding)
        self.global_output.append(output)
        #self.global_y.append(data.y)

        # print(model.prototypes.weight.size())

        # 💡 --> from the PGCLModel class: adapt forward method

        #_embedding, _output = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
        _embedding, _output = self.model(data_aug)

        embedding = torch.cat((embedding, _embedding))
        output = torch.cat((output, _output))
        
        """
        :: START CRITERION ? ::
        
        loss = 0
        hard_q, neg_q, z = [], [], []
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if self.queue is not None:
                    if self.use_queue or not torch.all(self.queue[i, -1, :] == 0):
                        # print("queue is not None")
                        self.use_queue = True
                        out = torch.cat((torch.mm(
                            self.queue[i],
                            self.model.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                #print(' {} queue : '.format(i), self.queue)
                #print(' out : ', out)
                # get assignments
                q = distributed_sinkhorn(out
                                         , epsilon=self.epsilon
                                         , world_size=self.world_size
                                         , sinkhorn_iterations=self.sinkhorn_iterations
                                        )[-bs:]
                
                #print(' q : ', q)
                # if not i:
                hard_q.append(torch.argmax(q, dim=1))
                ##################### important hyperparameter #####################
                neg_q.append(torch.argsort(q, dim=1)[:, 2:8]) # optional choices
                if not i:
                    pass#global_plabel.append(hard_q[0])
            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / self.temperature
                z.append(embedding[bs * v: bs * (v + 1), :] / self.temperature)
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))

        contrast_loss = self.loss_cal(#self.model,
            z[0], z[1], hard_q, neg_q
            #, hard_selection=self.hard_selection, sample_reweighting=self.sample_reweighting
        )
        #print('out : ', out)
        #print('queue : ', self.queue.shape)
        #print('subloss : ', subloss)
        #print('contrast_loss : ', contrast_loss)
        
        #print('embedding : ', embedding)
        #print('output : ', output)
        #print(' q : ', q)
        #print(' out : ', out)
        #print('z : ', z)
        
        loss += contrast_loss + 6 * subloss / (np.sum(self.nmb_crops) - 1)
        
        
        :: END CRITERION ? ::
        """
        loss = self.criterion(
            bs,
            crops_for_assign = self.crops_for_assign,
            nmb_crops = self.nmb_crops,
            queue = self.queue,
            prototypes = self.model.prototypes,
            output = output,
            embedding = embedding,
        )

        #loss_all += loss.item() #* data.num_graphs
        
        return loss
        