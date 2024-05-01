import numpy as np


class Accuracy:
    '''Class for calculating accuracy.'''
    def calculate(self, predictions, y):  # 0 for train data
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        comparisons = predictions == y
        accuracy = np.mean(comparisons)
        self.cum_accuracy += np.sum(comparisons)
        self.cum_count += len(comparisons)
        return accuracy

    def calc_cum(self):
        return self.cum_accuracy / self.cum_count

    def reset_cum(self):
        self.cum_accuracy = 0
        self.cum_count = 0
        
        
    def calculate1(self, predictions, y):  # 1 for validation data
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        comparisons = predictions == y
        accuracy = np.mean(comparisons)
        self.cum_accuracy1 += np.sum(comparisons)
        self.cum_count1 += len(comparisons)
        return accuracy
    
    def calc_cum1(self):
        return self.cum_accuracy1 / self.cum_count1
    
    def reset_cum1(self):
        self.cum_accuracy1 = 0
        self.cum_count1 = 0