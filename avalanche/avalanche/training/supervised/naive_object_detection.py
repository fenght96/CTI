################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-02-2022                                                             #
# Author(s): Lorenzo Pellegrini, Antonio Carta                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from typing import Optional, Sequence

import torch
from pkg_resources import parse_version
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.benchmarks.utils.data_loader import detection_collate_fn, \
    TaskBalancedDataLoader, detection_collate_mbatches_fn
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
import itertools
from typing import Any, Dict, List, Tuple, Union
from torchvision.transforms import functional as F
from torch import device

class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32, device=torch.device("cpu"))
        else:
            tensor = tensor.to(torch.float32)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor



class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        if not isinstance(image_size, torch.Tensor):  # could be a tensor in tracing
            for i in instance_lists[1:]:
                assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

class ObjectDetectionTemplate(SupervisedTemplate):
    """
    The object detection naive strategy.

    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine-tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.

    This strategy can be used as a template for any object detection strategy.
    This template assumes that the provided model follows the same interface
    of torchvision detection models.

    For more info, refer to "TorchVision Object Detection Finetuning Tutorial":
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    """

    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: int = 1,
            device="cpu",
            plugins: Optional[Sequence["SupervisedPlugin"]] = None,
            evaluator: EvaluationPlugin = default_evaluator,
            eval_every=-1,
            peval_mode="epoch",
            scaler=None):
        """
        Creates a naive detection strategy instance.

        :param model: The PyTorch detection model. This strategy accepts model
            from the torchvision library (as well as all model sharing the same
            interface/behavior)
        :param optimizer: PyTorch optimizer.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        :param scaler: The scaler from PyTorch Automatic Mixed Precision
            package. More info here: https://pytorch.org/docs/stable/amp.html.
            Defaults to None.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode
        )
        self.scaler = scaler  # torch.cuda.amp.autocast scaler
        """
        The scaler from PyTorch Automatic Mixed Precision package.
        More info here: https://pytorch.org/docs/stable/amp.html
        """

        # Object Detection attributes
        self.detection_loss_dict = None
        """
        A dictionary of detection losses.

        Only valid during the training phase.
        """

        self.detection_predictions = None
        self.mb_output = None
        """
        A list of detection predictions.

        This is different from mb_output: mb_output is a list of dictionaries 
        (one dictionary for each image in the input minibatch), 
        while this field, which is populated after calling `criterion()`,
        will be a dictionary {image_id: list_of_predictions}.

        Only valid during the evaluation phase. 
        """

    def make_train_dataloader(
            self, num_workers=0, shuffle=True, pin_memory=True,
            persistent_workers=False, **kwargs):
        """Data loader initialization.

        Called at the start of each learning experience after the dataset
        adaptation.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        :param persistent_workers: If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            Used only if `PyTorch >= 1.7.0`.
        """

        other_dataloader_args = {}

        if parse_version(torch.__version__) >= parse_version('1.7.0'):
            other_dataloader_args['persistent_workers'] = persistent_workers

        self.dataloader = TaskBalancedDataLoader(
            self.adapted_dataset,
            oversample_small_groups=True,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_mbatches=detection_collate_mbatches_fn,
            collate_fn=detection_collate_fn,
            **other_dataloader_args
        )

    def make_eval_dataloader(self, num_workers=0, pin_memory=True, **kwargs):
        """
        Initializes the eval data loader.
        :param num_workers: How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process.
            (default: 0).
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        :param kwargs:
        :return:
        """
        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=self.eval_mb_size,
            pin_memory=pin_memory,
            collate_fn=detection_collate_fn
        )

    def criterion(self):
        """
        Compute the loss function.

        The initial loss dictionary must be obtained by first running the
        forward pass (the model will return the detection_loss_dict).
        This function will only obtain a single value.

        Beware that the loss can only be obtained for the training phase as no
        loss dictionary is returned when evaluating.
        """
        if self.is_training:
            return sum(
                loss for loss in self.detection_loss_dict.values())
        else:
            # eval does not compute the loss directly.
            # Metrics will use self.mb_output and self.detection_predictions
            # to compute AP, AR, ...
            self.detection_predictions = \
                {target["image_id"].item(): output
                 for target, output in zip(self.mb_y, self.mb_output)}
            return torch.zeros((1,))

    def forward(self):
        """
        Compute the model's output given the current mini-batch.

        For the training phase, a loss dictionary will be returned.
        For the evaluation phase, this will return the model predictions.
        """
        if self.is_training:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                self.detection_loss_dict = self.model(self.mb_x, self.mb_y)
            return self.detection_loss_dict
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            outs = self.model(self.mb_x)
            return [{k: v.to('cpu') for k, v in t.items()} for t in outs]

    def _unpack_minibatch(self):
        # Unpack minibatch mainly takes care of moving tensors to devices.
        # In addition, it will prepare the targets in the proper dict format.
        # images = list(image.to(self.device) for image in self.mbatch[0])
        # targets = [{k: v.to(self.device) for k, v in t.items()}
        #            for t in self.mbatch[1]]
        # self.mbatch[0] = images
        # self.mbatch[1] = targets
        pass

    def backward(self):
        if self.scaler is not None:
            self.scaler.scale(self.loss).backward()
        else:
            self.loss.backward()

    def optimizer_step(self, **kwargs):
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

class ODetecronTemplate(ObjectDetectionTemplate):
    """
    The object detection naive strategy.

    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine-tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.

    This strategy can be used as a template for any object detection strategy.
    This template assumes that the provided model follows the same interface
    of torchvision detection models.

    For more info, refer to "TorchVision Object Detection Finetuning Tutorial":
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    """

    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: int = 1,
            device="cpu",
            plugins: Optional[Sequence["SupervisedPlugin"]] = None,
            evaluator: EvaluationPlugin = default_evaluator,
            eval_every=-1,
            peval_mode="epoch",
            scaler=None):
        """
        Creates a naive detection strategy instance.

        :param model: The PyTorch detection model. This strategy accepts model
            from the torchvision library (as well as all model sharing the same
            interface/behavior)
        :param optimizer: PyTorch optimizer.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        :param scaler: The scaler from PyTorch Automatic Mixed Precision
            package. More info here: https://pytorch.org/docs/stable/amp.html.
            Defaults to None.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
            scaler=scaler,
        )




    def criterion(self):
        """
        Compute the loss function.

        The initial loss dictionary must be obtained by first running the
        forward pass (the model will return the detection_loss_dict).
        This function will only obtain a single value.

        Beware that the loss can only be obtained for the training phase as no
        loss dictionary is returned when evaluating.
        """
        if self.is_training:
            return sum(
                loss for loss in self.detection_loss_dict.values())
        else:
            # eval does not compute the loss directly.
            # Metrics will use self.mb_output and self.detection_predictions
            # to compute AP, AR, ...
            self.detection_predictions = \
                {target["image_id"].item(): output
                 for target, output in zip(self.mb_y, self.mb_output)}
            return torch.zeros((1,))

    def forward(self):
        """
        Compute the model's output given the current mini-batch.

        For the training phase, a loss dictionary will be returned.
        For the evaluation phase, this will return the model predictions.
        """
        input_batchs = self._minibatch_detectron()
        if self.is_training:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                self.detection_loss_dict = self.model(input_batchs)
            return self.detection_loss_dict
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            preds = self.model(input_batchs)
            # print(preds)

            outs = []
            for pred in preds:
                temp = {}
                temp['scores'] = pred['instances'].scores.to('cpu')
                temp['labels'] = pred['instances'].pred_classes.to('cpu')
                temp['boxes'] = pred['instances'].pred_boxes.tensor.to('cpu')
                temp['boxes'][:,2] = temp['boxes'][:,2] - temp['boxes'][:,0]
                temp['boxes'][:,3] = temp['boxes'][:,3] - temp['boxes'][:,1]
                print(temp['boxes'])
                outs.append(temp)
            return outs

    # def _unpack_minibatch(self):
    #     # Unpack minibatch mainly takes care of moving tensors to devices.
    #     # In addition, it will prepare the targets in the proper dict format.
    #     pass
    
    def _minibatch_detectron(self):
        '''
            input_batchs : [{str:tensor}]
            image, file_name,image_id,instances,height,width
        '''
        input_batchs = []
        for i in range(len(self.mbatch[0])):
            input_batch = {}
            w,h = F.get_image_size(self.mbatch[0][i])
            input_batch['image'] = self.mbatch[0][i]#F.pil_to_tensor(self.mbatch[0][i])
            input_batch['file_name'] = self.mbatch[1][i]['image_id']
            input_batch['instances'] = Instances([w,h])
            
            input_batch['height'] = h
            input_batch['width'] = w
            input_batch['instances'].set('gt_boxes',Boxes(self.mbatch[1][i]['boxes']))
            input_batch['instances'].set('gt_classes',self.mbatch[1][i]['labels'])
            input_batchs.append(input_batch)
        return input_batchs





__all__ = [
    'detection_collate_fn',
    'ObjectDetectionTemplate',
    'ODetecronTemplate'
]
