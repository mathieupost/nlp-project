from sklearn.pipeline import make_pipeline, make_union

from model.base import AbstractPredictor


class ShowDownPredictor(AbstractPredictor):

    def __init__(self, transforms, classifier):
        self.transforms = transforms

        union = make_union(*[t() for t in transforms])
        pipeline = [union]
        self.pipeline = make_pipeline(*pipeline)
        self.classifier = classifier
