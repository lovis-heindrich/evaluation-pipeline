import numpy as np
import matplotlib.pyplot as plt
import pickle
import torchvision

def gridshow(img, normalize=False):
    # For input range [-1,1]
    if normalize:
        img = img / 2 + 0.5
    imshow(torchvision.utils.make_grid(img))

def imshow(img, normalize=False):
    # For input range [-1,1]
    if normalize:
        img = img / 2 + 0.5
    npimg = img.to("cpu").numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def save_results(res, path):
    with open(path, 'wb') as f:
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    
def load_results(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
        return res

def plot_accuracy(vals, labels, title):
    plt.figure()
    plt.style.use('seaborn-darkgrid')
    x = range(1, len(vals[0])+1)
    plt.xticks(ticks=x)
    for val, label in zip(vals, labels):
        plt.plot(x, val, lw=2, label=label)
    plt.xlim([1, 5])
    plt.ylim([0, 1.05])
    plt.xlabel("Level")
    plt.ylabel("Percent")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.show()

def plot_accuracy_comparison(res, selected=["Accuracy", "Activation"]):
    plt.figure(figsize=(14, 6))
    plt.style.use('seaborn-darkgrid')
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    subplots = [ax1, ax2]
    cols = list(res.columns.levels[0])
    #rows = list(res.index.levels[0])
    rows = selected
    x = range(1, 6)
    for col, subplt in zip(cols, subplots):
        subplt.set_xticks(ticks=x)
        subplt.set_xlim([1, 5])
        subplt.set_ylim([0, 1.05])
        subplt.set_xlabel("Level of difficulty")
        subplt.set_ylabel("Percent")
        subplt.title.set_text(col)
        for row in rows:
            value = res.loc[row, (col, "Mean")]
            std = res.loc[row, (col, "Std")]
            subplt.plot(x, value, lw=2, label=row)
            subplt.fill_between(x, y1=value-std, y2=value+std, alpha=.3)
        subplt.legend(loc="lower left")
    plt.show()
