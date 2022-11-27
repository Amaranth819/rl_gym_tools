import numpy as np
from collections import defaultdict
from base_policy import BasePolicy
from logger import Logger
from collector import Collector

'''
    Reference: https://github.com/thu-ml/tianshou/blob/master/tianshou/trainer/base.py
'''
class BaseTrainer(object):
    def __init__(self, 
        model : BasePolicy,
        collector : Collector,
        epochs : int = 100,
        batch_size : int = 256,
        log_path = './log/',
        best_model_path = None,
        eval_frequency = None,
        eval_epochs = 1,
        steps_per_collect = 1000,
        update_per_step = 1,
        reset_buffer_every_collect = False,
    ) -> None:
        self.model = model
        self.collector = collector
        self.epochs = epochs
        self.batch_size = batch_size
        self.log = Logger(log_path)
        self.best_model_path = best_model_path
        self.eval_frequency = eval_frequency
        self.eval_epochs = eval_epochs
        self.steps_per_collect = steps_per_collect
        self.update_per_step = update_per_step
        self.reset_buffer_every_collect = reset_buffer_every_collect
        
        self.best_test_reward = float('-inf')
        self.total_steps = 0


    def _policy_update_step(self):
        raise NotImplementedError


    def _train_step(self, curr_epoch):
        train_collect_result = self.collector.collect(self.steps_per_collect, 1, is_training = True, reset_buffer = self.reset_buffer_every_collect)
        update_result = self._policy_update_step()
        train_result = {**train_collect_result, **update_result}
        self.log.add(curr_epoch, train_result, 'Train/')

        self.total_steps += train_collect_result['n_steps']
        print('####################')
        print('# Epoch: {:d}'.format(curr_epoch))
        print('# Total steps: {:d}'.format(self.total_steps))
        for tag, scalar_val in train_result.items():
            print('# {:s}: {:.5f}'.format(tag, scalar_val))
        print('####################\n')


    def _test_step(self, curr_epoch):
        test_collect_result = self.collector.collect(self.steps_per_collect, self.eval_epochs, is_training = False, reset_buffer = False)
        self.log.add(curr_epoch, test_collect_result, 'Test/')
        print('Epoch {:d} Eval: Rewards = {:.2f} +- {:.2f} | Steps = {:.2f} +- {:.2f}'.format(
            curr_epoch, 
            test_collect_result['Sample Rewards Mean'],
            test_collect_result['Sample Rewards Std'],
            test_collect_result['Sample Steps Mean'],
            test_collect_result['Sample Steps Std'],
        ))
        if test_collect_result['Sample Rewards Mean'] > self.best_test_reward:
            self.best_test_reward = test_collect_result['Sample Rewards Mean']
            print('Get a better model!\n')
            if self.best_model_path is not None:
                self.model.save(self.best_model_path)
        else:
            print('Don\'t get a better model!\n')


    def learn(self):
        for e in np.arange(self.epochs) + 1:
            self._train_step(e)
            if self.eval_frequency is not None and e % self.eval_frequency == 0:
                self._test_step(e)
            self.model.step_lr_scheduler()




class OffPolicyTrainer(BaseTrainer):
    def __init__(
        self, 
        model: BasePolicy, 
        collector: Collector, 
        epochs: int = 100, 
        batch_size: int = 256, 
        log_path='./log/', 
        best_model_path=None, 
        eval_frequency=None, 
        eval_epochs = 1,
        steps_per_collect=1000, 
        update_per_step=1
    ) -> None:
        super().__init__(model, collector, epochs, batch_size, log_path, best_model_path, eval_frequency, eval_epochs, steps_per_collect, update_per_step, reset_buffer_every_collect = False)


    def _policy_update_step(self):
        train_log_info = defaultdict(lambda: [])
        for _ in range(int(self.steps_per_collect * self.update_per_step)):
            batch = self.collector.buffer.sample(self.batch_size)
            batch_train_log_info = self.model.learn(batch)
            for tag, val in batch_train_log_info.items():
                train_log_info[tag].append(val)

        for tag, val_list in train_log_info.items():
            train_log_info[tag] = np.mean(val_list)

        return train_log_info