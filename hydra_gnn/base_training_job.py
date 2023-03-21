from hydra_gnn.models import HomogeneousNetwork, HeterogeneousNetwork, HeterogeneousNeuralTreeNetwork, HomogeneousNeuralTreeNetwork
from hydra_gnn.mp3d_dataset import Hydra_mp3d_htree_data
import sys
import time
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tensorboardX import SummaryWriter


class BaseTrainingJob:
    def __init__(self, dataset_dict, network_params, double_precision=False):
        # data_type: (1)homogeneous (2)heterogeneous (3)homogeneous_htree (4)heterogeneous_htree
        data_type = dataset_dict['train'].data_type().split('_')
        self._network_type = 'neural_tree' if len(data_type) == 2 else 'baseline'
        self._graph_type = data_type[0]
        self._dataset_dict = dataset_dict

        # initialize training parameters
        self._training_params = self.create_default_params()
        self._update_training_params(network_params=network_params)
        
        input_dim = dataset_dict['train'].get_data(0).num_node_features()
        
        if self._graph_type == 'homogeneous':
            self._update_training_params(network_params={'input_dim': input_dim,
                'output_dim': dataset_dict['train'].get_data(0).num_room_labels()})
        else:
            self._update_training_params(network_params={'input_dim_dict': input_dim,
                'output_dim': dataset_dict['train'].get_data(0).num_room_labels()})
    
        # initialize network
        self.clean_up_network_params()
        self._net = self.initialize_network()
        if double_precision:
            self._net.double()

    @staticmethod
    def create_default_params():
        network_params = {'hidden_dim': 32,
                          'num_layers': 3,
                          'dropout': 0.25,
                          'conv_block': 'GraphSAGE',
                          'GAT_hidden_dims': 16,
                          'GAT_heads': [4, 4],
                          'GAT_concats': [True, False],
                          'ignored_label': 25}
        optimization_params = {'lr': 0.01,
                               'num_epochs': 200,
                               'weight_decay': 0.001,
                               'batch_size': 64,
                               'shuffle': True}
        training_params = {'network_params': network_params, 
                           'optimization_params': optimization_params}
        return training_params

    def clean_up_network_params(self):
        if self._training_params['network_params']['conv_block'][:3] == 'GAT':
            self._training_params['network_params'].pop('num_layers')
            self._training_params['network_params'].pop('hidden_dim')
        else:
            self._training_params['network_params'].pop('GAT_hidden_dims')
            self._training_params['network_params'].pop('GAT_heads')
            self._training_params['network_params'].pop('GAT_concats')

    def print_training_params(self, f=sys.stdout):
        for params, params_dict in self._training_params.items():
            print(params, file=f)
            for param_name, value in params_dict.items():
                print('   {}: {}'.format(param_name, value), file=f)

    def _update_training_params(self, network_params=None, optimization_params=None):
        if network_params is not None:
            for key in network_params:
                self._training_params['network_params'][key] = network_params[key]
        if optimization_params is not None:
            for key in optimization_params:
                self._training_params['optimization_params'][key] = optimization_params[key]

    def initialize_network(self):
        if self._graph_type == 'homogeneous':
            if self._network_type == 'baseline':
                return HomogeneousNetwork(**self._training_params['network_params'])
            else:
                return HomogeneousNeuralTreeNetwork(**self._training_params['network_params'])
        else:
            if self._network_type == 'baseline':
                return HeterogeneousNetwork(**self._training_params['network_params'])
            else:
                return HeterogeneousNeuralTreeNetwork(**self._training_params['network_params'])
    
    def train_job_type(self):
        return f"{self._graph_type} {self._network_type}"

    def get_dataset(self, split_name):
        return self._dataset_dict[split_name]

    def ignored_label(self):
        return self._training_params['network_params']['ignored_label']

    def train(self, log_folder, optimization_params=None, decay_epochs=100, decay_rate=1.0,
              early_stop_window=-1, min_log_epoch=0, verbose=False, gpu_index=0):
        # update parameters
        self._update_training_params(optimization_params=optimization_params)
        optimization_params = self._training_params['optimization_params']

        # move training to gpu if available
        assert gpu_index >= 0
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            device = torch.device("cpu")
        elif gpu_index < num_gpus:
            # gpu_mem = ((i, torch.cuda.mem_get_info(device=i)[0]) for i in range(num_gpus))
            # device = torch.device(f"cuda:{max(gpu_mem, key=lambda x: x[1])[0]}")
            device = torch.device(f"cuda:{gpu_index}")
        else:
            device = torch.device("cuda:0")
        self._net.to(device)
        if device == "cpu":
            print("Warning: not training on GPU!")
        else:
            print(f"Training on GPU {device.index}.")

        # create data loader
        train_loader = DataLoader(self.get_dataset('train'), batch_size=optimization_params['batch_size'],
            shuffle=optimization_params['shuffle'])
        val_loader = DataLoader(self.get_dataset('val'), batch_size=optimization_params['batch_size'],
            shuffle=optimization_params['shuffle'])
        test_loader = DataLoader(self.get_dataset('test'), batch_size=optimization_params['batch_size'],
            shuffle=optimization_params['shuffle'])

        # set up optimizer
        opt = optim.Adam(self._net.parameters(), lr=optimization_params['lr'], 
            weight_decay=optimization_params['weight_decay'])
        my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=decay_epochs, gamma=decay_rate)

        # train
        max_val_acc = 0
        best_model_state = None
        writer = SummaryWriter(log_folder)

        tic = time.perf_counter()
        early_stop_step = 0
        for epoch in range(optimization_params['num_epochs']):
            early_stop_step += 1
            total_loss = 0.
            num_train_labels = 0
            self._net.train()
            for batch in train_loader:
                opt.zero_grad()

                pred_vec = self._net(batch.to(device))
                if self._graph_type == 'homogeneous':
                    label = batch.y[batch.room_mask]
                else:
                    if self._network_type == 'baseline':
                        label = batch['rooms'].y
                    else:
                        label = batch['room_virtual'].y
                mask = (label != self._training_params['network_params']['ignored_label'])
                loss = self._net.loss(pred_vec, label, mask)
                loss.backward()
                opt.step()

                total_loss += loss.item()
                num_train_labels += mask.sum().item()
            total_loss /= num_train_labels
            writer.add_scalar('loss', total_loss, epoch)

            # writer.add_scalar('lr', opt.param_groups[0]["lr"], epoch)
            my_lr_scheduler.step()

            if verbose:
                train_result = self.test(train_loader)
                writer.add_scalar('train result', train_result, epoch)

            # validation and testing
            val_result = self.test(val_loader)
            writer.add_scalar('validation result', val_result, epoch)
            if epoch >= min_log_epoch and val_result > max_val_acc:
                max_val_acc = val_result
                best_model_state = deepcopy(self._net.state_dict())
                early_stop_step = 0
            if verbose and (epoch + 1) % 10 == 0:
                print('Epoch {:03}. Loss: {:.4f}. Train accuracy: {:.4f}. Validation accuracy: {:.4f}.'
                        .format(epoch, total_loss, train_result, val_result))
            if early_stop_step == early_stop_window and epoch > early_stop_window:
                if verbose:
                    print('Early stopping condition reached at {} epoch.'.format(epoch))
                break

        toc = time.perf_counter()
        print('Training completed (time elapsed: {:.4f} s). '.format(toc - tic))
        info = {'training_time': toc - tic, 'num_epochs': epoch + 1}

        # load best model to compute test result
        self._net.load_state_dict(best_model_state)
        tic = time.perf_counter()
        test_result = self.test(test_loader)
        toc = time.perf_counter()
        print('Testing completed (time elapsed: {:.4f} s). '.format(toc - tic))
        print('Best validation accuracy: {:.4f}, corresponding test accuracy: {:.4f}.'.
                format(max_val_acc, test_result))
        info['test_time'] = toc - tic

        torch.cuda.empty_cache()

        return self._net, (max_val_acc, test_result), info

    def test(self, data_loader, get_per_label_accuracy=False):
        self._net.eval()
        device = next(self._net.parameters()).device

        correct = 0
        total = 0
        if get_per_label_accuracy:
            num_valid_rooms = self._training_params['network_params']['output_dim'] - 1
            accuracy_matrix = np.zeros((num_valid_rooms, num_valid_rooms + 1), dtype=int)
        for batch in data_loader:
            with torch.no_grad():
                pred = self._net(batch.to(device)).argmax(dim=1)
                if self._graph_type == 'homogeneous':
                    label = batch.y[batch.room_mask]
                else:
                    if self._network_type == 'baseline':
                        label = batch['rooms'].y
                    else:
                        label = batch['room_virtual'].y
                mask = (label != self._training_params['network_params']['ignored_label'])

                pred = pred[mask]
                label = label[mask]

            correct += pred.eq(label).sum().item()
            total += torch.numel(label)
            if get_per_label_accuracy:
                for l in range(num_valid_rooms):
                    if l == self._training_params['network_params']['ignored_label']:
                        continue
                    # # number of label i; number of true positive; number of false positive
                    # label_l = (label == l)
                    # pred_l = (pred == l)
                    # accuracy_matrix[l, 0] = label_l.sum().item()
                    # accuracy_matrix[l, 1] = (label_l & pred_l).sum().item()
                    # accuracy_matrix[l, 2] = ((~label_l) & pred_l).sum().item()
                    for ll in range(num_valid_rooms + 1):
                        accuracy_matrix[l, ll] = (pred[label == l] == ll).sum().item()

        if get_per_label_accuracy:
            return correct / total, accuracy_matrix
        else:
            return correct / total

    def test_individual_graph(self, dataset, model=None):
        if model is not None:
            self._net = model

        self._net.eval()
        device = next(self._net.parameters()).device

        output_list = []  # list of (correct, total) tuple
        for data in dataset:
            with torch.no_grad():
                pred = self._net(data.to(device)).argmax(dim=1)
                if self._graph_type == 'homogeneous':
                    label = data.y[data.room_mask]
                else:
                    if self._network_type == 'baseline':
                        label = data['rooms'].y
                    else:
                        label = data['room_virtual'].y
                mask = (label != self._training_params['network_params']['ignored_label'])

                pred = pred[mask]
                label = label[mask]
                output_list.append((pred.eq(label).sum().item(), torch.numel(label)))

        return output_list
