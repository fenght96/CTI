################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""
    Avalanche supports data loading using pytorch's dataloaders.
    This module provides custom dataloaders for continual learning such as
    support for balanced dataloading between different tasks or balancing
    between the current data and the replay memory.
"""
import itertools
from collections import defaultdict
from itertools import chain
from typing import Dict, Sequence

import torch
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader

from avalanche.benchmarks.utils import AvalancheDataset


def _default_collate_mbatches_fn(mbatches):
    """Combines multiple mini-batches together.

    Concatenates each tensor in the mini-batches along dimension 0 (usually this
    is the batch size).

    :param mbatches: sequence of mini-batches.
    :return: a single mini-batch
    """
    batch = []
    for i in range(len(mbatches[0])):
        t = torch.cat([el[i] for el in mbatches], dim=0)
        batch.append(t)
    return batch


def detection_collate_fn(batch):
    """
    Collate function used when loading detection datasets using a DataLoader.
    """
    return tuple(zip(*batch))


def detection_collate_mbatches_fn(mbatches):
    """
    Collate function used when loading detection datasets using a DataLoader.
    """
    lists_dict = defaultdict(list)
    for mb in mbatches:
        for mb_elem_idx, mb_elem in enumerate(mb):
            lists_dict[mb_elem_idx].append(mb_elem)

    lists = []
    for mb_elem_idx in range(max(lists_dict.keys()) + 1):
        lists.append(list(itertools.chain.from_iterable(
            lists_dict[mb_elem_idx]
        )))

    return lists


class TaskBalancedDataLoader:
    """Task-balanced data loader for Avalanche's datasets."""

    def __init__(
        self,
        data: AvalancheDataset,
        oversample_small_tasks: bool = False,
        collate_mbatches=_default_collate_mbatches_fn,
        **kwargs
    ):
        """Task-balanced data loader for Avalanche's datasets.

        The iterator returns a mini-batch balanced across each task, which
        makes it useful when training in multi-task scenarios whenever data is
        highly unbalanced.

        If `oversample_small_tasks == True` smaller tasks are
        oversampled to match the largest task. Otherwise, once the data for a
        specific task is terminated, that task will not be present in the
        subsequent mini-batches.

        :param data: an instance of `AvalancheDataset`.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """
        self.data = data
        self.dataloaders: Dict[int, DataLoader] = {}
        self.oversample_small_tasks = oversample_small_tasks
        self.collate_mbatches = collate_mbatches

        # split data by task.
        task_datasets = []
        for task_label in self.data.task_set:
            tdata = self.data.task_set[task_label]
            task_datasets.append(tdata)

        # the iteration logic is implemented by GroupBalancedDataLoader.
        # we use kwargs to pass the arguments to avoid passing the same
        # arguments multiple times.
        if "data" in kwargs:
            del kwargs["data"]
        # needed if they are passed as positional arguments
        kwargs["oversample_small_groups"] = oversample_small_tasks
        kwargs["collate_mbatches"] = collate_mbatches
        self._dl = GroupBalancedDataLoader(datasets=task_datasets, **kwargs)

    def __iter__(self):
        for el in self._dl.__iter__():
            yield el

    def __len__(self):
        return self._dl.__len__()


class GroupBalancedDataLoader:
    """Data loader that balances data from multiple datasets."""

    def __init__(
        self,
        datasets: Sequence[AvalancheDataset],
        oversample_small_groups: bool = False,
        collate_mbatches=_default_collate_mbatches_fn,
        batch_size: int = 32,
        **kwargs
    ):
        """Data loader that balances data from multiple datasets.

        Mini-batches emitted by this dataloader are created by collating
        together mini-batches from each group. It may be used to balance data
        among classes, experiences, tasks, and so on.

        If `oversample_small_groups == True` smaller groups are oversampled to
        match the largest group. Otherwise, once data from a group is
        completely iterated, the group will be skipped.

        :param datasets: an instance of `AvalancheDataset`.
        :param oversample_small_groups: whether smaller groups should be
            oversampled to match the largest one.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param batch_size: the size of the batch. It must be greater than or
            equal to the number of groups.
        :param kwargs: data loader arguments used to instantiate the loader for
            each group separately. See pytorch :class:`DataLoader`.
        """
        self.datasets = datasets
        self.dataloaders = []
        self.oversample_small_groups = oversample_small_groups
        self.collate_mbatches = collate_mbatches

        # check if batch_size is larger than or equal to the number of datasets
        assert batch_size >= len(datasets)

        # divide the batch between all datasets in the group
        ds_batch_size = batch_size // len(datasets)
        remaining = batch_size % len(datasets)

        for data in self.datasets:
            bs = ds_batch_size
            if remaining > 0:
                bs += 1
                remaining -= 1
            self.dataloaders.append(DataLoader(data, batch_size=bs, **kwargs))
        self.max_len = max([len(d) for d in self.dataloaders])

    def __iter__(self):
        iter_dataloaders = []
        for dl in self.dataloaders:
            iter_dataloaders.append(iter(dl))

        max_num_mbatches = max([len(d) for d in iter_dataloaders])
        for it in range(max_num_mbatches):
            mb_curr = []
            is_removed_dataloader = False
            # copy() is necessary because we may remove keys from the
            # dictionary. This would break the generator.
            for tid, t_loader in enumerate(iter_dataloaders):
                try:
                    batch = next(t_loader)
                except StopIteration:
                    # StopIteration is thrown if dataset ends.
                    if self.oversample_small_groups:
                        # reinitialize data loader
                        iter_dataloaders[tid] = iter(self.dataloaders[tid])
                        batch = next(iter_dataloaders[tid])
                    else:
                        # We iteratated over all the data from this group
                        # and we don't need the iterator anymore.
                        iter_dataloaders[tid] = None
                        is_removed_dataloader = True
                        continue
                mb_curr.append(batch)
            yield self.collate_mbatches(mb_curr)

            # clear empty data-loaders
            if is_removed_dataloader:
                while None in iter_dataloaders:
                    iter_dataloaders.remove(None)

    def __len__(self):
        return self.max_len


class GroupBalancedInfiniteDataLoader:
    """Data loader that balances data from multiple datasets emitting an
    infinite stream."""

    def __init__(
        self,
        datasets: Sequence[AvalancheDataset],
        collate_mbatches=_default_collate_mbatches_fn,
        **kwargs
    ):
        """Data loader that balances data from multiple datasets emitting an
        infinite stream.

        Mini-batches emitted by this dataloader are created by collating
        together mini-batches from each group. It may be used to balance data
        among classes, experiences, tasks, and so on.

        :param datasets: an instance of `AvalancheDataset`.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param kwargs: data loader arguments used to instantiate the loader for
            each group separately. See pytorch :class:`DataLoader`.
        """
        self.datasets = datasets
        self.dataloaders = []
        self.collate_mbatches = collate_mbatches

        for data in self.datasets:
            infinite_sampler = RandomSampler(
                data, replacement=True, num_samples=10 ** 10
            )
            dl = DataLoader(data, sampler=infinite_sampler, **kwargs)
            self.dataloaders.append(dl)
        self.max_len = 10 ** 10

    def __iter__(self):
        iter_dataloaders = []
        for dl in self.dataloaders:
            iter_dataloaders.append(iter(dl))

        while True:
            mb_curr = []
            for tid, t_loader in enumerate(iter_dataloaders):
                batch = next(t_loader)
                mb_curr.append(batch)
            yield self.collate_mbatches(mb_curr)

    def __len__(self):
        return self.max_len


class ReplayDataLoader:
    """Custom data loader for rehearsal/replay strategies."""

    def __init__(self, data: AvalancheDataset, memory: AvalancheDataset = None,
                 oversample_small_tasks: bool = False,
                 collate_mbatches=_default_collate_mbatches_fn,
                 batch_size: int = 32,
                 batch_size_mem: int = 32,
                 task_balanced_dataloader: bool = False,
                 **kwargs):
        """ Custom data loader for rehearsal strategies.

        The iterates in parallel two datasets, the current `data` and the
        rehearsal `memory`, which are used to create mini-batches by
        concatenating their data together. Mini-batches from both of them are
        balanced using the task label (i.e. each mini-batch contains a balanced
        number of examples from all the tasks in the `data` and `memory`).

        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        :param data: AvalancheDataset.
        :param memory: AvalancheDataset.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param batch_size: the size of the data batch. It must be greater
            than or equal to the number of tasks.
        :param batch_size_mem: the size of the memory batch. If
            `task_balanced_dataloader` is set to True, it must be greater than
            or equal to the number of tasks.
        :param task_balanced_dataloader: if true, buffer data loaders will be
            task-balanced, otherwise it creates a single data loader for the
            buffer samples.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """

        self.data = data
        self.memory = memory
        self.loader_data: Sequence[DataLoader] = {}
        self.loader_memory: Sequence[DataLoader] = {}
        self.oversample_small_tasks = oversample_small_tasks
        self.collate_mbatches = collate_mbatches

        num_keys = len(self.memory.task_set)
        if task_balanced_dataloader:
            assert batch_size_mem >= num_keys, \
                "Batch size must be greator or equal " \
                "to the number of tasks in the memory " \
                "and current data."

        # Create dataloader for data items
        self.loader_data, _ = self._create_dataloaders(
            data, batch_size, 0, False, **kwargs)

        # Create dataloader for memory items
        if task_balanced_dataloader:
            single_group_batch_size = batch_size_mem // num_keys
            remaining_example = batch_size_mem % num_keys
        else:
            single_group_batch_size = batch_size_mem
            remaining_example = 0

        self.loader_memory, remaining_example = self._create_dataloaders(
            memory, single_group_batch_size, remaining_example,
            task_balanced_dataloader, **kwargs)

        self.max_len = max([len(d) for d in chain(
            self.loader_data.values(), self.loader_memory.values())]
                           )

    def __iter__(self):
        iter_data_dataloaders = {}
        iter_buffer_dataloaders = {}

        for t in self.loader_data.keys():
            iter_data_dataloaders[t] = iter(self.loader_data[t])
        for t in self.loader_memory.keys():
            iter_buffer_dataloaders[t] = iter(self.loader_memory[t])

        max_len = max([len(d) for d in iter_data_dataloaders.values()])

        try:
            for it in range(max_len):
                mb_curr = []
                self._get_mini_batch_from_data_dict(
                    self.data, iter_data_dataloaders,
                    self.loader_data, False,
                    mb_curr)

                self._get_mini_batch_from_data_dict(
                    self.memory, iter_buffer_dataloaders,
                    self.loader_memory, self.oversample_small_tasks,
                    mb_curr)

                yield self.collate_mbatches(mb_curr)
        except StopIteration:
            return

    def __len__(self):
        return self.max_len

    def _get_mini_batch_from_data_dict(self, data, iter_dataloaders,
                                       loaders_dict, oversample_small_tasks,
                                       mb_curr):
        # list() is necessary because we may remove keys from the
        # dictionary. This would break the generator.
        for t in list(iter_dataloaders.keys()):
            t_loader = iter_dataloaders[t]
            try:
                tbatch = next(t_loader)
            except StopIteration:
                # StopIteration is thrown if dataset ends.
                # reinitialize data loader
                if oversample_small_tasks:
                    # reinitialize data loader
                    iter_dataloaders[t] = iter(loaders_dict[t])
                    tbatch = next(iter_dataloaders[t])
                else:
                    del iter_dataloaders[t]
                    continue
            mb_curr.append(tbatch)

    def _create_dataloaders(self, data_dict, single_exp_batch_size,
                            remaining_example, task_balanced_dataloader,
                            **kwargs):
        loaders_dict: Dict[int, DataLoader] = {}
        if task_balanced_dataloader:
            for task_id in data_dict.task_set:
                data = data_dict.task_set[task_id]
                current_batch_size = single_exp_batch_size
                if remaining_example > 0:
                    current_batch_size += 1
                    remaining_example -= 1
                loaders_dict[task_id] = DataLoader(
                    data, batch_size=current_batch_size, **kwargs)
        else:
            loaders_dict[0] = DataLoader(
                data_dict, batch_size=single_exp_batch_size, **kwargs)

        return loaders_dict, remaining_example



class ReplayDetDataLoader:
    """Custom data loader for rehearsal/replay strategies."""

    def __init__(self, data: AvalancheDataset, memory: AvalancheDataset = None,
                 oversample_small_tasks: bool = False,
                 collate_mbatches=_default_collate_mbatches_fn,
                 batch_size: int = 32,
                 task_balanced_dataloader: bool = False,
                 **kwargs):
        """ Custom data loader for rehearsal strategies.

        The iterates in parallel two datasets, the current `data` and the
        rehearsal `memory`, which are used to create mini-batches by
        concatenating their data together. Mini-batches from both of them are
        balanced using the task label (i.e. each mini-batch contains a balanced
        number of examples from all the tasks in the `data` and `memory`).

        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        :param data: AvalancheDataset.
        :param memory: AvalancheDataset.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param batch_size: the size of the data batch. It must be greater
            than or equal to the number of tasks.
        :param batch_size_mem: the size of the memory batch. If
            `task_balanced_dataloader` is set to True, it must be greater than
            or equal to the number of tasks.
        :param task_balanced_dataloader: if true, buffer data loaders will be
            task-balanced, otherwise it creates a single data loader for the
            buffer samples.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """

        self.data = data
        self.memory = memory
        self.loader_data: Sequence[DataLoader] = {}
        self.loader_memory: Sequence[DataLoader] = {}
        self.oversample_small_tasks = oversample_small_tasks
        self.collate_mbatches = collate_mbatches
        self.batch_size = batch_size


        # Create dataloader for data items
        self.loader_data, _ = self._create_dataloaders(
            data, 1, 0, False, **kwargs)



        self.loader_memory, remaining_example = self._create_dataloaders(
            memory, 1, 0,
            task_balanced_dataloader, **kwargs)

        max_len = max(len(d) for d in self.loader_data.values()) // (self.batch_size - 2)
        max_len_mem = max(len(d) for d in self.loader_memory.values()) // 2
        self.max_len = max(max_len, max_len_mem)

    def __iter__(self):
        iter_data_dataloaders = {}
        iter_buffer_dataloaders = {}

        for t in self.loader_data.keys():
            iter_data_dataloaders[t] = iter(self.loader_data[t])
        for t in self.loader_memory.keys():
            iter_buffer_dataloaders[t] = iter(self.loader_memory[t])

        max_len = max([len(d)  for d in iter_data_dataloaders.values()]) // (self.batch_size - 2)
        max_len_data = sum([len(d) for d in iter_buffer_dataloaders.values()]) //2
        max_len = max(max_len, max_len_data)

        try:
            for it in range(max_len):
                mb_curr = []
                for _ in range(self.batch_size - 2):
                    self._get_mini_batch_from_data_dict(
                        self.data, iter_data_dataloaders,
                        self.loader_data, False,
                        mb_curr)
                for _ in range(2):
                    self._get_mini_batch_from_data_dict(
                        self.memory, iter_buffer_dataloaders,
                        self.loader_memory, self.oversample_small_tasks,
                        mb_curr)
                for x in mb_curr:
                    x  = x[1]
                    for k,v in x.items():
                        x[k] = v.squeeze(0)
                temp_img = [x[0].squeeze(0) for x in mb_curr]
                temp_label = [x[1] for x in mb_curr]
                temp_task_id = [x[-1].squeeze(0) for x in mb_curr]
                mb_curr = [temp_img, temp_label, temp_task_id]
                yield mb_curr
        except StopIteration:
            return

    def __len__(self):
        return self.max_len 

    def _get_mini_batch_from_data_dict(self, data, iter_dataloaders,
                                       loaders_dict, oversample_small_tasks,
                                       mb_curr):
        # list() is necessary because we may remove keys from the
        # dictionary. This would break the generator.
        for t in list(iter_dataloaders.keys()):
            t_loader = iter_dataloaders[t]
            try:
                tbatch = next(t_loader)
            except StopIteration:
                # StopIteration is thrown if dataset ends.
                # reinitialize data loader
                if oversample_small_tasks:
                    # reinitialize data loader
                    iter_dataloaders[t] = iter(loaders_dict[t])
                    tbatch = next(iter_dataloaders[t])
                else:
                    del iter_dataloaders[t]
                    continue
            mb_curr.append(tbatch)

    def _create_dataloaders(self, data_dict, single_exp_batch_size,
                            remaining_example, task_balanced_dataloader,
                            **kwargs):
        loaders_dict: Dict[int, DataLoader] = {}
        if task_balanced_dataloader:
            for task_id in data_dict.task_set:
                data = data_dict.task_set[task_id]
                current_batch_size = single_exp_batch_size
                if remaining_example > 0:
                    current_batch_size += 1
                    remaining_example -= 1
                loaders_dict[task_id] = DataLoader(
                    data, batch_size=current_batch_size, **kwargs)
        else:
            loaders_dict[0] = DataLoader(
                data_dict, batch_size=single_exp_batch_size, **kwargs)

        return loaders_dict, remaining_example

__all__ = [
    "detection_collate_fn",
    "detection_collate_mbatches_fn",
    "TaskBalancedDataLoader",
    "GroupBalancedDataLoader",
    "ReplayDataLoader",
    "GroupBalancedInfiniteDataLoader",
    'ReplayDetDataLoader',
]
