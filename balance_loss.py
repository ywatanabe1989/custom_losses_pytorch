def balance_loss(loss, Tb, counts_arr=None, n_classes_int=None):
  if not 'torch' in dir():
    import torch

  # define n_classes_int
  if not n_classes_int:
    try:
      n_classes_int = len(counts_arr)
    except:
      n_classes_int = int(Tb.max())+1

  # define counts_arr
  try:
    _ = counts_arr.shape
    counts_arr = torch.Long(counts_arr)
  except:
    counts_arr = torch.zeros(n_classes_int)
    for i in range(n_classes_int):
      counts_arr[i] += (Tb == i).sum()

  weights = torch.zeros_like(Tb, dtype=torch.float)
  # recip_probs = np.zeros_like(Tb, dtype=np.float)
  probs_arr = counts_arr / counts_arr.sum()
  recip_probs_arr = probs_arr ** (-1)
  for i in range(n_classes_int):
    mask = (Tb == i)
    weights[mask] += recip_probs_arr[i]
  weights_norm = (weights / weights.mean()).to(loss.dtype).to(loss.device)
  loss *= weights_norm
  return loss
  
if __name__ == '__main__'
  ## TEST ##
  def test(loss, Tb, counts_arr=None, n_classes_int=None, title=None):
    balanced_loss = balance_loss(loss, Tb, counts_arr=counts_arr, n_classes_int=n_classes_int)
    print()
    print(title)
    print('batched targets: {}'.format(Tb))
    print('loss_orig: {}'.format(loss))
    print('balanced_loss: {}'.format(balanced_loss))
    if loss.mean() == balanced_loss.mean():
      print('balanced_loss.mean() is the same as loss.mean()')
    print()

  loss = torch.Tensor([1,1,1,1,1,1])
  Tb = torch.LongTensor([0,0,0,1,1,2])

  test(loss, Tb, title='w/o counts nor n_classes_int')
  test(loss, Tb, counts_arr=np.array([1, 2, 3]), title='w/ counts, w/o n_classes_int')
  test(loss, Tb, n_classes_int=4, title='w/ n_classes_int, w/o  counts_arr')
  test(loss, Tb, n_classes_int=4, counts_arr=np.array([1,2,3,4]), title='w/ both n_classes_int and counts_arr')
  ##########
