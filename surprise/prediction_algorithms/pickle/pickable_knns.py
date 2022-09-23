from .pickable_mixin import PickableMixin
from surprise.prediction_algorithms import KNNBasic, KNNWithMeans, KNNBaseline, KNNWithZScore


class PickableKNNBasic(PickableMixin, KNNBasic):
    pass

class PickableKNNWithMeans(PickableMixin, KNNWithMeans):
    pass

class PickableKNNBaseline(PickableMixin, KNNBaseline):
    pass

class PickableKNNWithZScore(PickableMixin, KNNWithZScore):
    pass

