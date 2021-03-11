from torch.utils.data import Dataset, DataLoader
import torchvision
import train
import utils
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
import seaborn as sns
from matplotlib import pyplot as plt

class TransformDS(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        return self.transform(image), label

class ClassificationPipeline():
    def __init__(self, device, testset, batch_size, num_classes=14):
        self.device = device
        self.testset = testset
        self.batch_size = batch_size
        self.num_classes = num_classes

    def run_test(self, classifier, transforms, showimage=True):
        res={"acc":[], "act_correct":[], "act_incorrect":[]}
        res_list = []
        act_all_list = []
        act_correct_list = []
        act_incorrect_list = []
        roc_list = []
        brier_list=[]
        img = []
        for i, t in enumerate(transforms):
            ds = TransformDS(self.testset, t)
            dl = DataLoader(ds, batch_size=self.batch_size, num_workers=0)
            
            # Run evaluation
            acc, act, metrics = train.test(classifier, dl, self.device, num_classes=self.num_classes)
            res["acc"].append(acc)
            res["act_correct"].append(metrics["activation_correct_mean"])
            res["act_incorrect"].append(metrics["activation_incorrect_mean"])
            res_list.append(acc)
            act_all_list.append(act.item())
            act_correct_list.append(metrics["activation_correct_mean"].item())
            act_incorrect_list.append(metrics["activation_incorrect_mean"].item())
            roc_list.append(metrics["roc"])
            brier_list.append(metrics["brier"])

            # Print test images
            if showimage:
                #print("Level "+ str(i+1))
                for counter, data in enumerate(dl):
                    x, y = data[0].to(self.device), data[1].to(self.device)
                    #utils.imshow(torchvision.utils.make_grid(x[:8,:,:,:], 8))
                    img.append(x[:8,:,:,:])
                    break

                print("Accuracy:", acc, "Correct:", metrics["n_correct"], "Incorrect:", metrics["n_incorrect"])
                print("ROC:", metrics["roc"], "PR IN:", metrics["pr_in"], "PR OUT:", metrics["pr_out"])
                print("Brier:", metrics["brier"])
                sns.distplot(metrics["activations_correct"], hist=False, label="Correct activations")
                sns.distplot(metrics["activations_incorrect"], hist=False, label="Incorrect activations")
                plt.show()
                plt.clf()

        # Res order - acc lvl 1-5, act lvl 1-5
        for act in act_all_list:
            res_list.append(act)
        for act in act_correct_list:
            res_list.append(act)
        for act in act_incorrect_list:
            res_list.append(act)
        for roc in roc_list:
            res_list.append(roc)
        for brier in brier_list:
            res_list.append(brier)
        if showimage:
            utils.gridshow(torch.cat(img))
        return res, res_list
    
    def get_test_images(self, n, transforms, showimage=True):
        img = []
        for t in transforms:
            ds = TransformDS(self.testset, t)
            dl = DataLoader(ds, batch_size=n, num_workers=0)
            # Print test images
            for i, data in enumerate(dl):
                x, y = data[0].to(self.device), data[1].to(self.device)
                img.append(x[:n,:,:,:])
                break
        if showimage:
            x = torch.cat(img)
            utils.imshow(torchvision.utils.make_grid(x, n))
        return img

    def average_run_tests(self, n, transforms, get_classifier, num_levels=5):
        run_list = []
        measures = ["Accuracy", "Activation", "Activation correct", "Activation incorrect", "ROC", "Brier"]
        levels = ["Level "+str(i+1) for i in range(num_levels)]

        level_index = levels*len(measures)
        measure_index = [item for sl in ([measure]*num_levels for measure in measures) for item in sl]
        
        for i in tqdm_notebook(range(n)):
            classifier = get_classifier()
            res, res_list = self.run_test(classifier, transforms, showimage=False)
            run_list.append(res_list)
        data = np.array(run_list)
        index_arr = [measure_index, level_index]
        #index_arr = [["Accuracy", "Accuracy", "Accuracy", "Accuracy", "Accuracy", "Correct/Total", "Correct/Total", "Correct/Total", "Correct/Total", "Correct/Total", "Activation", "Activation", "Activation", "Activation", "Activation", "Activation Correct", "Activation Correct", "Activation Correct", "Activation Correct", "Activation Correct", "Activation Incorrect", "Activation Incorrect", "Activation Incorrect", "Activation Incorrect", "Activation Incorrect"],
        #            ["Level 1","Level 2","Level 3","Level 4","Level 5", "Level 1","Level 2","Level 3","Level 4","Level 5", "Level 1","Level 2","Level 3","Level 4","Level 5","Level 1","Level 2","Level 3","Level 4","Level 5","Level 1","Level 2","Level 3","Level 4","Level 5"]]
        index_tpl = list(zip(*index_arr))
        index = pd.MultiIndex.from_tuples(index_tpl, names=["Challenge level", "Measure"])
        df = pd.DataFrame(data, columns=index)
        return (df.agg([np.mean, np.std])).T, df