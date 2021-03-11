import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ood_loader
from classification import TransformDS
from tqdm import tqdm_notebook

class ChallengeDataset():
    def __init__(self, dataset, name, transform=None, generator=None, gan=False):
        self.name = name
        self.dataset = dataset
        self.dataloader = None
        self.results = []
        self.transform = transform

        # Gan parameters
        self.generator = generator
        self.gan = gan

    # Handles regeneration of ood examples in case of GAN generators
    def get_dataset(self):
        if not self.gan:
            return self.dataset
        else:
            with torch.no_grad():
                return self.generator.get_gan_examples()

class OodPipeline():
    def __init__(self, ood_factor, testset, device):
        self.testlen = len(testset)
        self.oodlen = self.testlen // ood_factor

        self.ood_label = torch.zeros(self.oodlen)
        self.indistribution_label = torch.ones(self.testlen)
        self.device = device

        self.indistribution_ds = ood_loader.OOD(testset, self.indistribution_label)

    def gen_ood_data(self, ood_ds):
        if len(ood_ds) < self.oodlen:
            print("ERROR IN OOD GENERATION; DATASET TOO SMALL")
        indices = torch.randperm(len(ood_ds))[:self.oodlen]
        subset = Subset(ood_ds, indices)
        ood_ds = ConcatDataset((self.indistribution_ds, subset))
        ood_dl = DataLoader(ood_ds, batch_size=len(ood_ds), num_workers=0, shuffle=True)
        return subset, ood_dl

    def gen_transformed_ood_data(self, ood_ds, transform):
        indices = torch.randperm(len(ood_ds))[:self.oodlen]
        subset = Subset(ood_ds, indices)
        ood_ds = ConcatDataset((self.indistribution_ds, subset))
        ds = TransformDS(ood_ds, transform)
        dl = DataLoader(ds, batch_size=len(ood_ds), num_workers=0, shuffle=True)
        return subset, dl

    def get_ood_label(self):
        return self.ood_label
    
    def two_color_hist(self, real_datapoints, fake_datapoints):
        bins = np.linspace(0, 1, 20)
        plt.hist(real_datapoints, bins, alpha=0.5, label='In-distribution')
        plt.hist(fake_datapoints, bins, alpha=0.5, label='Out-of-distribution')
        plt.legend(loc='upper left')
        plt.show()
    
    def plot_curve(self, x, y, title, xlabel, ylabel, plot_baseline=True):
        plt.figure()
        plt.plot(x, y, color='red', lw=2)
        if plot_baseline:
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        #plt.legend(loc="lower right")
        plt.show()

    def run_evaluation(self, classifier, testset, plot = False):
        for i, data in enumerate(testset):
            x, y = data[0].detach().to(self.device), data[1].to(self.device)
            with torch.no_grad():
                softmax, max_activation, predicted = classifier.get_prediction(x)

            real_datapoints = max_activation[y == 1]
            fake_datapoints = max_activation[y == 0]
            avg_real = torch.mean(real_datapoints).item()
            avg_fake = torch.mean(fake_datapoints).item()

            # Point plot with dots for in / out of distribution
            if plot:
                self.two_color_hist(real_datapoints.to("cpu").numpy(), fake_datapoints.to("cpu").numpy())

            # ROC calculation with sklearn
            fpr, tpr, thresholds = roc_curve(y.to("cpu").numpy(), max_activation.to("cpu").numpy())
            roc_score = roc_auc_score(y.to("cpu").numpy(), max_activation.to("cpu").numpy())
            if plot:
                self.plot_curve(fpr, tpr, "AUROC" ,"False positive rate", "True positive rate")

            # PR calculation with sklearn
            # In-distribution is positive
            precision, recall, thresholds = precision_recall_curve(y.to("cpu").numpy(), max_activation.to("cpu").numpy())
            pr_in_score = average_precision_score(y.to("cpu").numpy(), max_activation.to("cpu").numpy())
            if plot:
                self.plot_curve(recall, precision, "AUPR OUT", "Recall"," Precision", plot_baseline = False)
            # Out-of-distribution is positive
            # Flip labels and invert softmax probabilities
            y_out = 1 - y
            activation_out = -1 * max_activation
            precision, recall, thresholds = precision_recall_curve(y_out.to("cpu").numpy(), activation_out.to("cpu").numpy())
            pr_out_score = average_precision_score(y_out.to("cpu").numpy(), activation_out.to("cpu").numpy())
            if plot:
                self.plot_curve(recall, precision, "AUPR OUT", "Recall"," Precision", plot_baseline = False)

            res = {"ROC": roc_score, "PR IN": pr_in_score, "PR OUT": pr_out_score, "MEAN REAL": avg_real, "MEAN FAKE": avg_fake}
            
            del data, x, y, softmax, max_activation, predicted
            torch.cuda.empty_cache()
            return res

    def make_df(self, res, name):
        df = pd.DataFrame(res)
        stat = df.agg([np.mean, np.std])
        stat = stat.T.stack().to_frame()
        stat.columns = [name]
        stat = stat.T.round(4)
        return stat

    def full_evaluation(self, get_classifier, challenge_data, n=10):
        # Refresh res for multiple runs
        for cd in challenge_data:
            cd.results = []
        for i in tqdm_notebook(range(n)):
            # Train classifier
            classifier = get_classifier()
            
            with torch.no_grad():
                for cd in challenge_data:
                    if cd.transform is not None:
                        cd.dataloader = self.gen_transformed_ood_data(cd.get_dataset(), cd.transform)[1]
                    else:
                        cd.dataloader = self.gen_ood_data(cd.get_dataset())[1]
                    res = self.run_evaluation(classifier, cd.dataloader)
                    del cd.dataloader
                    cd.results.append(res)

                del classifier
                torch.cuda.empty_cache()
        res_list = [(ds.results, ds.name) for ds in challenge_data]
        res = [self.make_df(x[0], x[1]) for x in res_list]
            
        df = pd.concat(res)
        return df, res_list