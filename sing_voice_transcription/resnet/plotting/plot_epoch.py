import matplotlib.pyplot as plt
import pickle

# PG learning curve
losses = pickle.load(open('./plotting/data/loss.pkl', 'rb'))
train_loss = losses['train']
valid_loss = losses['valid']

plt.plot(train_loss, label='Training Loss')
plt.plot(valid_loss, label='Validation Loss')
plt.title('Loss v.s. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./plotting/results/learning_curve.png')
