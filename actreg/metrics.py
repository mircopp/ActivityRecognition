import numpy as np

class ScoreMap():

    def __init__(self, categories, scores, strategy='kubic', factor=1, bias=0):
        """
        Initilizes the ScoreMap.
        :param categories: The categories of the map as a feature vector
        :param scores: The scores corresponding to the categoeries as a feature vector
        :param strategy: The strategy of how to rank the scores (linear -> default, quadratic, kubic, exponential)
        :param factor: The factor of the model multiplied with the score after strategy calculation.
        :param bias: The bias added to the result.
        """

        self.fit(categories, scores)
        self.ranking_strategy = strategy
        self.factor = factor
        self.bias = bias

    def fit(self, categories, scores):
        """
        Fits the categories to the given scores.
        :param categories: The categories of the map as a feature vector
        :param scores: The scores corresponding to the categoeries as a feature vector
        :return: The resulting ScoreMap
        """

        self.categorial_map = {}
        self.score_map = {}
        for i in range(len(categories)):
            self.categorial_map[i] = categories[i]
            self.score_map[i] = scores[i]
        return self

    def get_activitiy(self, activity_number):
        """
        Returns the corresponding activity of a given index number
        :param activity_number: Index of the category
        :return: The activity description
        """

        return self.categorial_map[activity_number]

    def get_score(self, activity_number):
        """
        Calculates the final score as bias + factor * strategy(initial_score(activity_number))
        :param activity_number: The index of the corresponding category
        :return: The final score
        """

        if self.ranking_strategy == 'kubic':
            return self.bias + self.factor * self.score_map[activity_number] ** 3
        elif self.ranking_strategy == 'quadratic':
            return self.bias + self.factor * self.score_map[activity_number] **2
        elif self.ranking_strategy == 'exponential':
            return self.bias + self.factor * np.exp(self.score_map[activity_number])
        else:
            # Default linear distribution
            return self.bias + self.factor * self.score_map[activity_number]