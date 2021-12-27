import numpy as np
from sklearn.metrics import confusion_matrix
import torch

def batch_iou(pred, target, ignore_index = None):
    try:
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    except:
        pass
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
    cfm = confusion_matrix(target, pred)
    s = cfm.shape[0]
    if s == 0:
        return 0
    _iou = 0
    for i in range(s):
        if cfm[i,i] == 0:
            continue
        _iou += cfm[i,i] / (np.sum(cfm[i,:]) + np.sum(cfm[:,i]) - cfm[i,i])
    return _iou / s

def cls_acc(logit, label):
        
    logit = logit.detach().sigmoid()
    logit = (logit >= 0.5)
    all_correct = torch.all(logit == label.byte(), dim=1).float().sum().item()
    return all_correct / logit.size(0)
    # pass

class Cls_Accuracy:
    def __init__(self, ):
        self.total = 0
        self.correct = 0
    
    def reset(self):
        self.total = 0
        self.correct = 0

    def add_batch(self, logit, label):
        
        logit = logit.sigmoid_()
        logit = (logit >= 0.5)
        all_correct = torch.all(logit == label.byte(), dim=1).float().sum().item()
        
        self.total += logit.size(0)
        self.correct += all_correct

    def evaluate(self):
        cls_acc = self.correct / self.total
        return dict(cls_acc = cls_acc)


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes, ignore_index = 255):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.ignore_index = ignore_index

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes) & (label_true != self.ignore_index) & (label_pred != self.ignore_index)
        
        hist = np.bincount(
            self.num_classes*label_true[mask] + label_pred[mask], 
            minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        
        return hist

    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))
        
        return {
            "Pixel_Accuracy": acc,
            "Mean_Accuracy": acc_cls,
            "Frequency_Weighted_IoU": fwavacc,
            "Mean_IoU": mean_iu,
            "Class_IoU": cls_iu,
        }
