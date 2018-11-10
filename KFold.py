import numpy as np


def generate_folds(self):
    total_idx = np.arange(self.X.shape[0])
    folds = np.array_split(total_idx, self.num_folds)
    for i in range(self.num_folds):
        f_tr = np.concatenate(
            [folds[j] for j in range(self.num_folds) if j != i])
        f_ev = folds[i]
        yield f_tr, f_ev

def kfolds_eval(self, *args, **kwargs):
    acc_arr = np.zeros(self.num_folds)
    for i, (idx_tr, idx_ev) in enumerate(self.generate_folds()):
        print('**** Current split {}'.format(i))
        X_tr, X_ev = self.X[idx_tr], self.X[idx_ev]
        y_tr, y_ev = self.y[idx_tr], self.y[idx_ev]

        self.model.train(X_tr, y_tr)
        y_pred = self.model.eval(X_ev, *args, **kwargs)

        acc_arr[i] = compute_accuracy(y_ev, y_pred)
        print('**** Current split acc {}'.format(acc_arr[i]))

    mean = acc_arr.mean()
    std = acc_arr.std()
    print('{} acc: {:.4f} +/- {:.4f} with kwargs: {}'.format(
        type(self.model).__name__, mean, std, kwargs))
    return mean


X = 10
y = 20
model = 5
num_folds = 5
