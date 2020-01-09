import tensorflow as tf
import torch


def macro_double_soft_f1_tf(y, y_hat): # Written in Tensorflow
    # https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost


def macro_double_soft_f1(y, y_hat, reduction='mean'): # Written in PyTorch
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (torch.FloatTensor): targets array of shape (BATCH_SIZE, N_LABELS), including 0. and 1.
        y_hat (torch.FloatTensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar): value of the cost function for the batch
    """

    # dtype = y_hat.dtype
    # y = y.to(dtype)

    # FloatTensor = torch.cuda.FloatTensor
    # y = FloatTensor(y)
    # y_hat = FloatTensor(y_hat)


    tp = (y_hat * y).sum(dim=0) # soft
    fp = (y_hat * (1-y)).sum(dim=0) # soft
    fn = ((1-y_hat) * y).sum(dim=0) # soft
    tn = ((1-y_hat) * (1-y)).sum(dim=0) # soft

    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0

    if reduction == 'none':
        return cost

    if reduction == 'mean':
        macro_cost = cost.mean()
        return macro_cost


if __name__ == '__main__':
    BS = 64
    y = np.random.randint(2, size=BS)
    y_hat = np.random.rand(BS)

    loss_tf = macro_double_soft_f1_tf(y, y_hat) # Written in Tensorflow
    loss_torch = macro_double_soft_f1(torch.FloatTensor(y), torch.FloatTensor(y_hat)) # Written in PyTorch

    print('Macro Double Soft F1 Loss')
    print('Written in Tensorflow :{}'.format(loss_tf))
    print('Written in Pytorch :{}'.format(loss_torch))
