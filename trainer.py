import tqdm
import os
from collections import defaultdict
import torch
import wandb


class Trainer:
    CHECKPOINTS_PATH = 'checkpoints'
    
    def __init__(self, model, criterion, metric, config, device='cuda'):
        self._model = model
        self._criterion = criterion
        self._metrics = metric
        self._device = device
        
        self._model.to(self._device)
        self._epochs = config['epochs'] 
        self._early_stopping = config['early_stopping']
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=config['lr'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            'max',
            factor=config['lr_reduce_rate'],
            patience=config['patience'],
            verbose=True
        )

        
        self._best_metric = float('-inf')
        if not os.path.exists(Trainer.CHECKPOINTS_PATH):
            os.makedirs(Trainer.CHECKPOINTS_PATH)
        
    
    def fit(self, train_loader, val_loader):
        passed_epochs_without_upgrades = 0
        
        wandb.watch(self._model, self._criterion, log='all', log_freq=10)
        for epoch in range(self._epochs):
            if passed_epochs_without_upgrades > self._early_stopping:
                return 
            
            self._model.train()
            train_metrics = self._run_epoch(epoch, train_loader, is_training=True)

            metrics_str = []
            for name, value in train_metrics.items():
                metrics_str.append(f'{name}: {float(value):.5f}')
            metrics_str = ' '.join(metrics_str)
            print('train metrics: ' + metrics_str)


            self._model.eval()
            val_metrics = self._run_epoch(epoch, train_loader, is_training=False)

            self._scheduler.step(val_metrics['dice'])
            print(self._optimizer.param_groups[0]['lr'])

            metrics_str = []
            for name, value in val_metrics.items():
                metrics_str.append(f'{name}: {float(value):.5f}')
            metrics_str = ' '.join(metrics_str)
            print('val metrics: ' + metrics_str)
            

            if self._best_metric < val_metrics['dice']:
                passed_epochs_without_upgrades = 0
                self._best_metric = val_metrics['dice']
                torch.save(self._model.state_dict(), os.path.join(Trainer.CHECKPOINTS_PATH, 'weights.pth'))
           
            
            passed_epochs_without_upgrades += 1
    
    def _run_epoch(self, epoch, loader, is_training):
        if is_training:
            pbar = tqdm.tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}')
        else:
            pbar = enumerate(loader)
        
        avg_metrics = defaultdict(float)
        for i, data in pbar:
            batch_metrics = self._step(data, is_training)
            for name, val in batch_metrics.items():
                avg_metrics[name] += val
        
        if not is_training:
            for name, val in avg_metrics.items():
                wandb.log({name: val / len(loader)})
        return {name: value / len(loader) for name, value in avg_metrics.items()}
    
    def _step(self, data, is_training=True):
        metrics_values = {}
        images = data['image'].to(self._device)
        y_true = data['mask'].float().unsqueeze(1).to(self._device)
        
        if is_training:
            self._optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            y_pred = self._model(images)
            loss = self._criterion(y_pred, y_true)
            
            for name, func in self._metrics:
                value = func(y_true=y_true, y_pred=torch.sigmoid(y_pred))
                metrics_values[name] = value.item()

            if is_training:
                loss.backward()
                self._optimizer.step()
        
        metrics_values['loss'] = loss.item()
        return metrics_values