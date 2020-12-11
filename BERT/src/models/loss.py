"""
This file handles the details of the loss function during training.
This includes: LossComputeBase and the standard NMTLossCompute, and sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable

from models.reporter import Statistics


def abs_loss(generator, symbols, vocab_size, device, train=True, label_smoothing=0.0):

    compute = NMTLossCompute(
        generator, symbols, vocab_size,
        label_smoothing=label_smoothing if train else 0.0)
    compute.to(device)
    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) : module that maps the output of the decoder to a distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) : torchtext vocab object representing the target output.
        normalzation (str): normalize by "sents" or "tokens".
    """

    def __init__(self, generator, pad_id):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.padding_idx = pad_id

    def _make_shard_state(self, batch, output,  attns=None):
        """
        Make shard state dictionary for shards() to return iterable shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.
        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, copy_params = None):
        """
        Compute the forward loss for the batch.
        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`): output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) : dictionary of attention distributions `[tgt_len x batch x src_len]`
          copy_params (params of train for pointer-generator and extractor) hybrid train.
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        shard_state = self._make_shard_state(batch, output, copy_params)
        output = shard_state['output']
        target = shard_state['target']
        if len(shard_state) > 4:
            copy_params_new = (shard_state['copy_params[0]'], shard_state['copy_params[1]'], shard_state['copy_params[2]'])
        else:
            copy_params_new = (shard_state['copy_params[0]'], shard_state['copy_params[1]'])
        if copy_params is not None:
            if len(copy_params) > 2:
                _, batch_stats = self._compute_loss(batch, output, target, g=copy_params_new[1], ext_dist=copy_params_new[0], ext_loss=copy_params_new[2])
            else:
                _, batch_stats = self._compute_loss(batch, output, target, g=copy_params_new[1], ext_dist=copy_params_new[0])
        else:
            _, batch_stats = self._compute_loss(batch, output, target)
        return batch_stats

    def sharded_compute_loss(self, batch, output, shard_size, normalization, copy_params = None):
        """
        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number
        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics
        """
        batch_stats = Statistics()
        shard_state = self._make_shard_state(batch, output, copy_params)
        for shard in shards(shard_state, shard_size):
            output = shard['output']
            target = shard['target']
            if copy_params is not None:
                g = shard['copy_params[1]']
                ext_dist = shard['copy_params[0]']
                if len(shard) > 2:
                    ext_loss = shard['copy_params[2]']
                if len(copy_params)>2:
                    loss, stats = self._compute_loss(batch, output, target, g, ext_dist, ext_loss)
                else:
                    loss, stats = self._compute_loss(batch, output, target, g, ext_dist)
            else:
                loss, stats = self._compute_loss(batch, output, target)
            (loss.div(float(normalization)) + ext_loss.mean() * 2).backward()
            batch_stats.update(stats)
        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets
        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing, KL-divergence between q_{smoothed ground truth prob.}(w) and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)
        return F.kl_div(output, model_prob, reduction='sum')


class KnowledgeDistillLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(KnowledgeDistillLoss, self).__init__()
        self.padding_idx = ignore_index

    def forward(self, output, target):
        loss = -(output * target).float()
        return loss.sum()


class PairwiseLoss(nn.Module):
    """
    The pairwise loss between the label and the prediction KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    Here we proposed the pairwise_loss that is needed to learn the relationship of sentence-level.
    """
    def __init__(self):
        super(PairwiseLoss, self).__init__()
        # self.loss = torch.nn.BCELoss(reduction='none')
        self.loss = torch.nn.MSELoss(reduction='none')

    def forward(self, output, target, mask):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        mask = mask.unsqueeze(1).float()
        mask = torch.bmm(mask.transpose(1,2), mask)  # The two dimensions of the matrix are multiplied.
        output_ = output.repeat(1, output.size(1)).reshape(output.size(0), output.size(1), output.size(1))
        outputt = output.unsqueeze(1).transpose(1,2)
        outputt = outputt.repeat(1,1,outputt.size(1))
        pairwise_output = nn.functional.sigmoid(5 * (outputt - output_)) * mask
        target1 = torch.zeros(mask.size()).to('cuda')
        for i in range(target1.size(0)):
            for j in range(target1.size(1)):
                for k in range(target1.size(2)):
                    if target[i][j] > target[i][k]:
                        target1[i][j][k] = 1
                    elif target[i][j] < target[i][k]:
                        target1[i][j][k] = 0
                    else:
                        target1[i][j][k] = 0.5
        target1 = target1 * mask
        half_mask = torch.ne(target1, 0.5).float()
        pairwise_output = pairwise_output * half_mask
        target1 = target1 * half_mask
        loss = self.loss(pairwise_output, target1) * mask
        return loss


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, symbols, vocab_size,
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, symbols['PAD'])
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _make_shard_state(self, batch, output, copy_params):
        if copy_params is not None:
            # print("len(copy params)")
            # print(len(copy_params))
            # exit()
            if len(copy_params) > 2:
                return {
                    "output": output,
                    "target": batch.tgt[:,1:],
                    "copy_params[0]": copy_params[0],
                    "copy_params[1]": copy_params[1],
                    "copy_params[2]": copy_params[2],
                }
            else:
                return {
                    "output": output,
                    "target": batch.tgt[:,1:],
                    "copy_params[0]": copy_params[0],
                    "copy_params[1]": copy_params[1],
                }
        else:
            return {
                "output": output,
                "target": batch.tgt[:,1:],
            }

    def _compute_loss(self, batch, output, target, g=None, ext_dist=None, ext_loss = None):
        bottled_output = self._bottle(output)
        scores = self.generator(bottled_output)
        if g is not None:
            scores = scores * self._bottle(g) + self._bottle(ext_dist)
        scores = torch.log(scores)
        gtruth =target.contiguous().view(-1)
        loss = self.criterion(scores, gtruth)
        if ext_loss is not None:
            stats = self._stats(loss.clone() + ext_loss.mean().clone() * 2, scores, gtruth)
        else:
            stats = self._stats(loss.clone(), scores, gtruth)
        return loss, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v
        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.
    Yields:
        Each yielded shard is a dict.
    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))
        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))
        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))
        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)

'''
class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, symbols, vocab_size,
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, symbols['PAD'])
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _make_shard_state(self, batch, output, copy_params):
        return {
            "output": output,
            "target": batch.tgt[:,1:],
            "copy_params[0]": copy_params[0],
            "copy_params[1]": copy_params[1],
        }

    def _compute_loss(self, batch, output, target, copy_params = None):
        # print("output")
        # print(output.size())
        bottled_output = self._bottle(output)
        # print("bottled_output ", bottled_output.size())
        # print("copy_params ", copy_params)
        # print(bottled_output)
        # exit()
        scores = self.generator(bottled_output)
        # print("scores ", scores.size())
        # print(scores)
        # exit()
        # print("scores ext : ", copy_params)


        if copy_params:
            # print("ex prob")
            # print(copy_params[0].size())
            # print("g")
            # print(copy_params[1].size())
            # new_scores = copy_params[1] * copy_params[0]
            # print("new_scores")
            # print(new_scores.size())
            # print("scores softmax: ", scores.size())

            # scores = scores * copy_params[1].view(-1,copy_params[1].size(2)) + copy_params[0].view(-1,copy_params[0].size(2))
            scores = torch.log(scores)


        gtruth =target.contiguous().view(-1)
        # print("scores logsoftmax: ", scores.size())
        # print("gtruth ", gtruth.size())
        # print(gtruth)
        # exit()



        loss = self.criterion(scores, gtruth)
        print('loss', loss.size())
        print(loss)

        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
'''
