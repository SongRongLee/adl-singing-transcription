import matplotlib.pyplot as plt
import pickle

# PG learning curve
losses = pickle.load(open('./plotting/data/loss.pkl', 'rb'))
train_epoch_id = [loss[0] for loss in losses['train']]
train_loss = [loss[1] for loss in losses['train']]
valid_epoch_id = [loss[0] for loss in losses['valid']]
valid_loss = [loss[1] for loss in losses['valid']]

plt.plot(train_epoch_id, train_loss, label='Training Loss')
plt.plot(valid_epoch_id, valid_loss, label='Validation Loss')
plt.title('Loss v.s. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./plotting/results/learning_curve.png')
