import math
import numpy as np
from torch.utils.data import Dataset

class InfoBatch(Dataset):
    def __init__(self, dataset, ratio = 0.5, Aratio=0.5, num_epoch=300, delta = 0.875):
        self.dataset = dataset
        self.ratio = ratio
        self.Aratio = Aratio
        self.num_epoch = num_epoch
        self.delta = delta
        self.iaug = self.dataset.iaug
        self.scores = np.ones([len(self.dataset)])
        # self.augsco = np.ones([len(self.dataset)*self.iaug])
        self.transform = dataset.transform
        self.weights = np.ones(len(self.dataset))
        self.augweig = np.ones([len(self.dataset)*self.iaug])
        self.save_num = 0

    def __setscore__(self, indices, values,n_iaug):
        bs = values.shape[0]//(1+n_iaug)
        # index_list=[i*self.iaug for i in range(bs)]
        self.scores[indices] = values[:bs].std(1) ### mean
        # self.scores[indices] = self.scores[indices]*0.99+0.01*values[:bs].mean(1)
        # self.scores[indices] = np.array([values[index_list[i]:index_list[i+1]].mean() for i in range(bs)])
        # for i in range(self.iaug):
        #     self.augsco[indices*self.iaug+i]=values[bs*(i+1):bs*(i+2)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        weight = self.weights[index]
        
        for i in range(self.iaug):
            weight = np.append(weight,self.augweig[index*self.iaug+i])
        return data, target, index, weight

    def prune(self):
        # prune samples that are well learned, rebalence the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance

        b = self.scores>self.scores.mean() # <
        # b1 = self.scores<self.scores.mean() # <
        well_learned_samples = np.where(b)[0]
        # well_learned_samples1 = np.where(b1)[0]
        pruned_samples = []
        pruned_samples.extend(np.where(np.invert(b))[0])
        # pruned_samples1 = []
        # pruned_samples1.extend(np.where(np.invert(b))[0])
        # selected = np.random.choice(well_learned_samples1, int(self.ratio*len(well_learned_samples))+len(pruned_samples1)-len(pruned_samples),replace=False)
        selected = np.random.choice(well_learned_samples, int(self.ratio*len(well_learned_samples)),replace=False)
        self.reset_weights()
        if len(selected)>0:
            self.weights[selected]= 1/self.ratio
            for i in range(self.iaug):
                self.augweig[selected*self.iaug+i]= 1/self.ratio 
            pruned_samples.extend(selected)
        print(f'Cut {len(self.dataset)-len(pruned_samples)} samples and {(len(self.dataset)-len(pruned_samples))*16} augmented samples for next iteration')
        self.save_num += len(self.dataset)-len(pruned_samples)
        np.random.shuffle(pruned_samples)

        return pruned_samples

    def pruning_sampler(self):
        return InfoBatchSampler(self, self.num_epoch, self.delta)

    def no_prune(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    # def mean_score(self):
    #     return self.scores.mean()

    # def normal_sampler_no_prune(self):
    #     return InfoBatchSampler(self.no_prune)

    # def get_weights(self,indexes):
    #     return self.weights[indexes]

    # def total_save(self):
    #     return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))
        self.augweig = np.ones([len(self.dataset)*self.iaug])



class InfoBatchSampler():
    def __init__(self, infobatch_dataset, num_epoch = math.inf, delta = 1):
        self.infobatch_dataset = infobatch_dataset
        self.seq = None
        self.stop_prune = num_epoch * delta
        self.seed = 0
        self.num_epoch = num_epoch
        self.delta = delta
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed+=1
        # if self.seed<=self.num_epoch//3:
        if self.seed<=self.num_epoch* (1-self.delta):
            self.seq = self.infobatch_dataset.no_prune()
        else:
            if self.seed>self.stop_prune:
                if self.seed <= self.stop_prune+1:
                    self.infobatch_dataset.reset_weights()
                self.seq = self.infobatch_dataset.no_prune()
            else:
                self.seq = self.infobatch_dataset.prune()
        # if self.seed>self.stop_prune:
        #     if self.seed <= self.stop_prune+1:
        #         self.infobatch_dataset.reset_weights()
        #     self.seq = self.infobatch_dataset.no_prune()
        # else:
        #     self.seq = self.infobatch_dataset.prune()
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self