import numpy as np

class ScoreMap():

    def __init__(self, categories, scores, strategy='kubic', factor=1, bias=0):
        self.fit(categories, scores)
        self.ranking_strategy = strategy
        self.factor = factor
        self.bias = bias

    def fit(self, categories, scores):
        self.categorial_map = {}
        self.score_map = {}
        for i in range(len(categories)):
            self.categorial_map[i] = categories[i]
            self.score_map[i] = scores[i]
        return self

    def get_activitiy(self, activity_number):
        return self.categorial_map[activity_number]

    def get_score(self, activity_number):
        if self.ranking_strategy == 'kubic':
            return self.bias + self.factor * self.score_map[activity_number] ** 3
        elif self.ranking_strategy == 'quadratic':
            return self.bias + self.factor * self.score_map[activity_number] **2
        elif self.ranking_strategy == 'exponential':
            return self.bias + self.factor * np.exp(self.score_map[activity_number])
        else:
            # Default linear distribution
            return self.bias + self.factor * self.score_map[activity_number]