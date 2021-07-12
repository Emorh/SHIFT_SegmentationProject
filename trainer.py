import tqdm
import os
import torch
import wandb

class Trainer:
    CHECKPOINTS_PATH = 'checkpoints'
    
    def __init__(self, model, criterion, metric, config, device='cuda'):
        self._model = model
        self._criterion = criterion
        self._metric_name = metric['name']
        self._metric = metric['func']
        self._device = device
        
        self._model.to(self._device)
        self._epochs = config['epochs'] 
        self._early_stopping = config['early_stopping']
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=config['lr'])
        
        self._best_metric = float('-inf')
        if not os.path.exists(Trainer.CHECKPOINTS_PATH):
            os.makedirs(Trainer.CHECKPOINTS_PATH)
        
    
    def fit(self, train_loader):
        passed_epochs_without_upgrades = 0
        
        wandb.watch(self._model, self._criterion, log='all', log_freq=10)
        for epoch in range(self._epochs):
            if passed_epochs_without_upgrades > self._early_stopping:
                return 
            
            self._model.train()
            train_metrics = self._run_epoch(epoch, train_loader, is_training=True)
            
            if self._best_metric < train_metrics[self._metric_name]:
                passed_epochs_without_upgrades = 0
                self._best_metric = train_metrics[self._metric_name]
                torch.save(self._model.state_dict(), os.path.join(Trainer.CHECKPOINTS_PATH, 'weights.pth'))
            
            
            print(f'Epoch {epoch}, loss: {train_metrics["loss"]}, \
                  {self._metric_name}: {train_metrics[self._metric_name]}')
            
            passed_epochs_without_upgrades += 1
    
    def _run_epoch(self, epoch, loader, is_training):
        if is_training:
            pbar = tqdm.tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}')
        else:
            pbar = enumerate(loader)
        
        avg_metrics = {'loss' : 0, self._metric_name : 0}
        for i, data in pbar:
            batch_metrics = self._step(data, is_training)
            avg_metrics['loss'] += batch_metrics['loss']
            avg_metrics[self._metric_name] += batch_metrics[self._metric_name]
        
        wandb.log({'loss': avg_metrics['loss'] / len(loader)})
        wandb.log({self._metric_name: avg_metrics[self._metric_name]/len(loader)})
        return {'loss': avg_metrics['loss'] / len(loader), 
                self._metric_name: avg_metrics[self._metric_name] / len(loader)}
    
    def _step(self, data, is_training=True):
        metrics_values = {}
        images = data['image'].to(self._device)
        y_true = data['mask'].float().unsqueeze(1).to(self._device)
        
        if is_training:
            self._optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            y_pred = self._model(images)
            loss = self._criterion(y_pred, y_true)
            
            metrics_values[self._metric_name] = \
                        self._metric(y_true=y_true, y_pred=torch.sigmoid(y_pred)).item()

            if is_training:
                loss.backward()
                self._optimizer.step()
        
        metrics_values['loss'] = loss.item()
        return metrics_values