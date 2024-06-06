from hydra_gnn.base_training_job import BaseTrainingJob
import time
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tensorboardX import SummaryWriter


class SemiSupervisedTrainingJob(BaseTrainingJob):
    def __init__(self, dataset, network_params, double_precision=False):
        # data_type: (1)homogeneous (2)heterogeneous (3)homogeneous_htree (4)heterogeneous_htree
        data_type = dataset.data_type().split("_")
        self._network_type = "neural_tree" if len(data_type) == 2 else "baseline"
        self._graph_type = data_type[0]
        self._dataset = dataset

        # initialize training parameters
        self._training_params = self.create_default_params()
        self._update_training_params(network_params=network_params)

        input_dim = dataset.get_data(0).num_node_features()
        if self._graph_type == "homogeneous":
            self._update_training_params(network_params={"input_dim": input_dim})
        else:
            self._update_training_params(network_params={"input_dim_dict": input_dim})

        if self._network_type == "baseline":
            self._update_training_params(
                network_params={
                    "output_dim_dict": {
                        "rooms": dataset.get_data(0).num_room_labels(),
                        "objects": dataset.get_data(0).num_object_labels(),
                    }
                }
            )
        else:
            self._update_training_params(
                network_params={
                    "output_dim_dict": {
                        "room": dataset.get_data(0).num_room_labels(),
                        "object": dataset.get_data(0).num_object_labels(),
                        "object-room": 1,
                        "room-room": 1,
                    }
                }
            )

        # initialize network
        self.clean_up_network_params()
        self._net = self.initialize_network()
        if double_precision:
            self._net.double()

    def train(
        self,
        log_folder,
        optimization_params=None,
        decay_epochs=100,
        decay_rate=1.0,
        early_stop_window=-1,
        min_log_epoch=0,
        verbose=False,
        gpu_index=0,
    ):
        # update parameters
        self._update_training_params(optimization_params=optimization_params)
        optimization_params = self._training_params["optimization_params"]

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
        data_loader = DataLoader(
            self._dataset,
            batch_size=optimization_params["batch_size"],
            shuffle=optimization_params["shuffle"],
        )
        test_loader = DataLoader(self._dataset, batch_size=len(self._dataset))

        # set up optimizer
        opt = optim.Adam(
            self._net.parameters(),
            lr=optimization_params["lr"],
            weight_decay=optimization_params["weight_decay"],
        )
        my_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=opt, step_size=decay_epochs, gamma=decay_rate
        )

        # train
        max_val_acc = 0
        best_model_state = None
        writer = SummaryWriter(log_folder)

        tic = time.perf_counter()
        early_stop_step = 0
        for epoch in range(optimization_params["num_epochs"]):
            early_stop_step += 1
            total_loss = 0.0
            self._net.train()
            for batch in data_loader:
                opt.zero_grad()

                pred_vec = self._net(batch.to(device))
                if self._graph_type == "homogeneous":
                    if self._network_type == "baseline":
                        label = batch.y[batch.room_mask], batch.y[~(batch.room_mask)]
                        mask = (
                            batch.train_mask[batch.room_mask],
                            batch.train_mask[~(batch.room_mask)],
                        )
                    else:
                        label = batch.y[batch.room_mask], batch.y[batch.object_mask]
                        mask = (
                            batch.train_mask[batch.room_mask],
                            batch.train_mask[batch.object_mask],
                        )
                else:
                    if self._network_type == "baseline":
                        label = batch["rooms"].y, batch["objects"].y
                        mask = batch["rooms"].train_mask, batch["objects"].train_mask
                    else:
                        label = batch["room_virtual"].y, batch["object_virtual"].y
                        mask = (
                            batch["room_virtual"].train_mask,
                            batch["object_virtual"].train_mask,
                        )
                loss = self._net.loss(pred_vec, label, mask)
                loss.backward()
                opt.step()

                total_loss += loss.item() * batch.num_graphs
            total_loss /= len(data_loader.dataset)
            writer.add_scalar("loss", total_loss, epoch)

            # writer.add_scalar('lr', opt.param_groups[0]["lr"], epoch)
            my_lr_scheduler.step()

            if verbose:
                train_result = self.test(test_loader, mask_name="train_mask")
                writer.add_scalar("train result", train_result, epoch)

            # validation and testing
            val_result = self.test(test_loader, mask_name="val_mask")
            writer.add_scalar("validation result", val_result, epoch)
            if epoch >= min_log_epoch and val_result > max_val_acc:
                max_val_acc = val_result
                best_model_state = deepcopy(self._net.state_dict())
                early_stop_step = 0
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    "Epoch {:03}. Loss: {:.4f}. Train accuracy: {:.4f}. Validation accuracy: {:.4f}.".format(
                        epoch, total_loss, train_result, val_result
                    )
                )
            if early_stop_step == early_stop_window and epoch > early_stop_window:
                if verbose:
                    print("Early stopping condition reached at {} epoch.".format(epoch))
                break

        toc = time.perf_counter()
        print("Training completed (time elapsed: {:.4f} s). ".format(toc - tic))
        info = {"training_time": toc - tic, "num_epochs": epoch + 1}

        # load best model to compute test result
        self._net.load_state_dict(best_model_state)
        tic = time.perf_counter()
        test_result = self.test(test_loader, mask_name="test_mask")
        toc = time.perf_counter()
        print("Testing completed (time elapsed: {:.4f} s). ".format(toc - tic))
        print(
            "Best validation accuracy: {:.4f}, corresponding test accuracy: {:.4f}.".format(
                max_val_acc, test_result
            )
        )
        info["test_time"] = toc - tic

        torch.cuda.empty_cache()

        return self._net, (max_val_acc, test_result), info

    def test(
        self, data_loader=None, mask_name="test_mask", get_type_separated_accuracy=False
    ):
        assert mask_name in ["train_mask", "val_mask", "test_mask"]
        if data_loader is None:
            data_loader = DataLoader(
                self._dataset,
                batch_size=self._training_params["optimization_params"]["batch_size"],
                shuffle=self._training_params["optimization_params"]["shuffle"],
            )

        self._net.eval()
        device = next(self._net.parameters()).device

        correct_room = 0
        correct_object = 0
        total_room = 0
        total_object = 0
        for batch in data_loader:
            with torch.no_grad():
                pred = self._net(batch.to(device))
                pred = (p.argmax(dim=1) for p in pred)
                if self._graph_type == "homogeneous":
                    if self._network_type == "baseline":
                        label = batch.y[batch.room_mask], batch.y[~(batch.room_mask)]
                        mask = (
                            batch[mask_name][batch.room_mask],
                            batch[mask_name][~(batch.room_mask)],
                        )
                    else:
                        label = batch.y[batch.room_mask], batch.y[batch.object_mask]
                        mask = (
                            batch[mask_name][batch.room_mask],
                            batch[mask_name][batch.object_mask],
                        )
                else:
                    if self._network_type == "baseline":
                        label = batch["rooms"].y, batch["objects"].y
                        mask = batch["rooms"][mask_name], batch["objects"][mask_name]
                    else:
                        label = batch["room_virtual"].y, batch["object_virtual"].y
                        mask = (
                            batch["room_virtual"][mask_name],
                            batch["object_virtual"][mask_name],
                        )
            batch_correct_room, batch_correct_object = (
                p[m].eq(l[m]).sum().item() for p, l, m in zip(pred, label, mask)
            )
            batch_totoal_room, batch_total_object = (
                torch.numel(l[m]) for l, m in zip(label, mask)
            )

            correct_room += batch_correct_room
            correct_object += batch_correct_object
            total_room += batch_totoal_room
            total_object += batch_total_object

        if get_type_separated_accuracy:
            return correct_room / total_room, correct_object / total_object
        else:
            return (correct_room + correct_object) / (total_room + total_object)

    def test_individual_graph(self, dataset, model=None):
        return NotImplemented
