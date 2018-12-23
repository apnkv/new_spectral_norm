import numpy as np
import torch


class VariableMonitor:
    def __init__(self, batch_var_mean=True):
        self.vars = []
        self.batches_processed = 0
        self.batch_var_mean = batch_var_mean

    def add_batch(self, var, batch_size):
        if type(var) == torch.Tensor:
            var = var.item()

        self.vars.append(var)
        self.batches_processed += batch_size

    def get_value(self):
        if self.batches_processed == 0:
            return 0

        vars_np = np.array(self.vars)

        if self.batch_var_mean:
            return np.mean(vars_np)
        else:
            return np.sum(vars_np) / self.batches_processed

    def reset(self):
        self.vars = []
        self.batches_processed = 0


class DistributionMonitor:
    def __init__(self):
        self.var_dist = None

    def add_batch(self, var):
        if type(var) == torch.Tensor:
            var = var.cpu().detach().numpy()
        
        assert(len(var.shape) == 1 or (len(var.shape) == 2 and var.shape[1] == 1))

        var = var.reshape(-1, 1)

        if self.var_dist is None:
            self.var_dist = var
        else:
            self.var_dist = np.concatenate([self.var_dist, var], axis=0)

    def get_dist(self):
        return self.var_dist
    
    def reset(self):
        self.var_dist = None


class MonitorsManager:
    def __init__(self, writer, batch_var_mean, hist, hist_mode, monitor_id=''):
        self.writer = writer
        self.batch_var_mean = batch_var_mean
        self.hist_mode = hist_mode

        self.hist = hist

        self.var_monitors_dict = {}
        self.dist_monitors_dict = {}

        self.monitor_id = monitor_id

        if len(self.monitor_id) > 0 and self.monitor_id[-1] != '/':
            self.monitor_id += '/'
    
    def add_batch(self, monitor_name, var, batch_size):
        self.var_monitors_dict[monitor_name].add_batch(var, batch_size)

    def add_batch_dict(self, stats_dict, batch_size):
        for monitor, value in stats_dict.items():
            if hasattr(value, 'shape') and len(value.shape) > 0:
                if self.hist:
                    if monitor not in self.dist_monitors_dict:
                        self.dist_monitors_dict[monitor] = DistributionMonitor()

                    self.dist_monitors_dict[monitor].add_batch(value)
            else:
                if monitor not in self.var_monitors_dict:
                    self.var_monitors_dict[monitor] = VariableMonitor(self.batch_var_mean)

                self.var_monitors_dict[monitor].add_batch(value, batch_size)

    def write(self, n_iter):
        for monitor_name, monitor in self.var_monitors_dict.items():
            self.writer.add_scalar(self.monitor_id + monitor_name, monitor.get_value(), n_iter)

        if self.hist:
            for montor_name, monitor in self.dist_monitors_dict.items():
                dist = monitor.get_dist()
                if dist is not None and len(dist) > 0:
                    self.writer.add_histogram(self.monitor_id + montor_name, np.array(monitor.get_dist()), n_iter, bins=self.hist_mode)

    def reset(self):
        for monitor in self.var_monitors_dict.values():
            monitor.reset()

        for monitor in self.dist_monitors_dict.values():
            monitor.reset()
