import torch
import numpy as np
from collections import defaultdict
from base_policy import BasePolicy
from logger import Logger
from collector import Collector


class BaseTrainer(object):
    def __init__(self, 
        collector : Collector,
        model : BasePolicy, 
        lr_scheduler_dict : 'dict[str, torch.optim.lr_scheduler._LRScheduler]' = None,
        epochs : int = 1000,
        batch_size : int = 128,
        log_path = './log/',
        best_model_path = None,
        eval_frequency = None,
        n_sample_steps = None,
        reset_buffer_every_collect = True
    ) -> None:
        self.collector = collector
        self.model = model
        self.lr_scheduler_dict = lr_scheduler_dict
        self.epochs = epochs
        self.batch_size = batch_size
        self.log = Logger(log_path)
        self.best_model_path = best_model_path
        self.eval_frequency = eval_frequency
        self.n_sample_steps = n_sample_steps
        self.reset_buffer_every_collect = reset_buffer_every_collect


    def _train(self):
        # Sampling trajectory
        eval_log_info = self.collector.collect(self.n_sample_steps, True, self.reset_buffer_every_collect)

        # Training
        train_log_info = defaultdict(lambda: [])
        for batch in self.collector.generate_data(self.batch_size, self.model.is_recurrent):
            batch_training_log_info = self.model.update(batch)

            for tag, val in batch_training_log_info.items():
                train_log_info[tag].append(val)

        # Summary
        for tag, val_list in train_log_info.items():
            train_log_info[tag] = np.mean(val_list)

        log_info = {**train_log_info, **eval_log_info}
        return log_info


    def learn(self):
        if self.epochs <= 0:
            print('No training task!')
            return

        # Eval before training
        if self.eval_frequency is not None:
            eval_log_info = self.collector.collect(self.n_sample_steps, True, self.reset_buffer_every_collect)
            best_eval_rewards_mean = eval_log_info['Sample Rewards Mean']
            self.log.add(0, eval_log_info, 'Eval/')
            print('Epoch 0 Eval: Rewards = {:.2f} +- {:.2f} | Steps = {:.2f} +- {:.2f}'.format(*eval_log_info.values()))

        # Training
        for e in np.arange(self.epochs) + 1:
            log_info = self._train()
            self.log.add(e, log_info, 'Train/')

            print('####################')
            print('# Epoch: %d' % e)
            for tag, scalar_val in log_info.items():
                print('# %s: %.5f' % (tag, scalar_val))
            print('####################\n')

            if self.lr_scheduler_dict is not None:
                for _, scheduler in self.lr_scheduler_dict.items():
                    scheduler.step()

            # Evaluate every certain epochs
            if self.eval_frequency is not None and e % self.eval_frequency == 0:
                eval_log_info = self.collector.collect(self.n_sample_steps, True, self.reset_buffer_every_collect)
                self.log.add(e, eval_log_info, 'Eval/')
                print('Epoch {:d} Eval: Rewards = {:.2f} +- {:.2f} | Steps = {:.2f} +- {:.2f}'.format(e, *eval_log_info.values()))

                if eval_log_info['Sample Rewards Mean'] > best_eval_rewards_mean:
                    print('Get a better model!\n')
                    best_eval_rewards_mean = eval_log_info['Sample Rewards Mean']
                    if self.best_model_path is not None:
                        self.model.save(self.best_model_path)
                else:
                    print('Don\'t get a better model!\n')