from hydra_gnn.models import HomogeneousNetwork, HeterogeneousNetwork, NeuralTreeNetwork
import sys
import time
from copy import deepcopy
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tensorboardX import SummaryWriter


class BaseTrainingJob:
    def __init__(self, network_type, dataset_dict, network_params, double_precision=False):
        assert network_type in ['homogeneous', 'heterogeneous', 'neural_tree']
        self._network_type = network_type
        self._dataset_dict = dataset_dict

        # initialize training parameters
        self._training_params = self.create_default_params()
        self.update_training_params(network_params=network_params)
        
        room_input_feature, object_input_feature = \
            dataset_dict['train'].get_data(0).num_node_features()
        assert room_input_feature == object_input_feature
        input_dim = object_input_feature
        self.update_training_params(network_params={'input_dim': input_dim,
            'output_dim': dataset_dict['train'].get_data(0).num_room_labels()})
    
        # initialize network
        self._net = self.initialize_network()
        if double_precision:
            self._net.double()

    @staticmethod
    def create_default_params():
        network_params = {'input_dim': None,
                          'output_dim': None,
                          'hidden_dim': 32,
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
        if self._training_params['network_params']['conv_block'] == 'GAT':
            delattr(self._training_params['network_params'],'num_layers' )
        else:
            delattr(self._training_params['network_params'],'GAT_hidden_dims' )
            delattr(self._training_params['network_params'],'GAT_heads' )
            delattr(self._training_params['network_params'],'GAT_concats' )

    def print_training_params(self, f=sys.stdout):
        for params, params_dict in self._training_params.items():
            print(params, file=f)
            for param_name, value in params_dict.items():
                print('   {}: {}'.format(param_name, value), file=f)

    def update_training_params(self, network_params=None, optimization_params=None):
        if network_params is not None:
            for key in network_params:
                self._training_params['network_params'][key] = network_params[key]
        if optimization_params is not None:
            for key in optimization_params:
                self._training_params['optimization_params'][key] = optimization_params[key]

    def initialize_network(self):
        if self._network_type == 'homogeneous':
            return HomogeneousNetwork(**self._training_params['network_params'])
        elif self._network_type == 'heterogeneous':
            return HeterogeneousNetwork(**self._training_params['network_params'])
        else:
            return NeuralTreeNetwork(**self._training_params['network_params'])

    def get_dataset(self, split_name):
        return self._dataset_dict[split_name]

    def train(self, log_folder, optimization_params=None, decay_epochs=100, decay_rate=1.0,
              early_stop_window=-1, verbose=False):
        # update parameters
        self.update_training_params(optimization_params=optimization_params)
        optimization_params = self._training_params['optimization_params']

        # move training to gpu if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._net.to(device)
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
        max_test_acc = 0  # If val_loader is not None, compute using the model weights that lead to best_val_acc
        best_model_state = None
        writer = SummaryWriter(log_folder)
        if val_loader is None:
            early_stop_window = -1  # do not use early stopping if there's no validation set

        tic = time.perf_counter()
        early_stop_step = 0
        for epoch in range(optimization_params['num_epochs']):
            early_stop_step += 1
            total_loss = 0.
            self._net.train()
            for batch in train_loader:
                opt.zero_grad()

                pred_vec = self._net(batch.to(device))
                if self._network_type == 'homogeneous':
                    label = batch.y[batch.room_mask]
                elif self._network_type == 'heterogeneous':
                    label = batch['rooms'].y
                else:
                    label = batch['room_virtual'].y
                mask = (label != self._training_params['network_params']['ignored_label'])
                loss = self._net.loss(pred_vec, label, mask)
                loss.backward()
                opt.step()

                total_loss += loss.item() * batch.num_graphs
            total_loss /= len(train_loader.dataset)
            writer.add_scalar('loss', total_loss, epoch)

            writer.add_scalar('lr', opt.param_groups[0]["lr"], epoch)
            my_lr_scheduler.step()

            if verbose:
                train_result = self.test(train_loader)
                writer.add_scalar('train result', train_result, epoch)

            # validation and testing
            if val_loader is not None:
                val_result = self.test(val_loader)
                writer.add_scalar('validation result', val_result, epoch)
                if val_result > max_val_acc:
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
            else:
                test_result = self.test(test_loader)
                writer.add_scalar('test result', test_result, epoch)
                if test_result > max_test_acc:
                    max_test_acc = test_result
                    best_model_state = deepcopy(self._net.state_dict())
                if verbose and (epoch + 1) % 10 == 0:
                    print('Epoch {:03}. Loss: {:.4f}. Train accuracy: {:.4f}. Test accuracy: {:.4f}.'.
                          format(epoch, total_loss, train_result, test_result))

        toc = time.perf_counter()
        print('Training completed (time elapsed: {:.4f} s). '.format(toc - tic))

        self._net.load_state_dict(best_model_state)

        if val_loader is not None:
            tic = time.perf_counter()
            test_result = self.test(test_loader)
            toc = time.perf_counter()
            print('Testing completed (time elapsed: {:.4f} s). '.format(toc - tic))
            print('Best validation accuracy: {:.4f}, corresponding test accuracy: {:.4f}.'.
                  format(max_val_acc, test_result))
            return self._net, (max_val_acc, test_result)
        else:
            print('Best test accuracy: {:.4f}.'.format(max_test_acc))
            return self._net, max_test_acc

    def test(self, data_loader):
        self._net.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        correct = 0
        total = 0
        for batch in data_loader:
            with torch.no_grad():
                pred = self._net(batch.to(device)).argmax(dim=1)
                if self._network_type == 'homogeneous':
                    label = batch.y[batch.room_mask]
                elif self._network_type == 'heterogeneous':
                    label = batch['rooms'].y
                else:
                    label = batch['room_virtual'].y
                mask = (label != self._training_params['network_params']['ignored_label'])

                pred = pred[mask]
                label = label[mask]

            correct += pred.eq(label).sum().item()
            total += torch.numel(label)

        return correct / total
