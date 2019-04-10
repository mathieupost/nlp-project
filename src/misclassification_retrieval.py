import numpy as np
import pandas as pd

from model.utils import get_dataset, split_data

from model.classifiers.classifier_showdown import ShowDownPredictor
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model.baseline.transforms import (
    RefutingWordsTransform,
    QuestionMarkTransform,
    HedgingWordsTransform,
    InteractionTransform,
    NegationOfRefutingWordsTransform,
    BoWTransform
)

from sklearn.neighbors import KNeighborsClassifier

from model.ext.transforms import (
    AlignedPPDBSemanticTransform,
    NegationAlignmentTransform,
    Word2VecSimilaritySemanticTransform,
    DependencyRootDepthTransform,
    SVOTransform,
)

training_data = get_dataset('url-versions-2015-06-14-clean-train.csv')
X, y = split_data(training_data)

predictor = ShowDownPredictor

transforms = {
        'Q': QuestionMarkTransform,
        'W2V': Word2VecSimilaritySemanticTransform,
        'PPDB': AlignedPPDBSemanticTransform,
        'NegAlgn': NegationAlignmentTransform,
        'RootDep': DependencyRootDepthTransform,
        'SVO': SVOTransform,
        'BoW' : BoWTransform,
}

clf = KNeighborsClassifier(18)

    #DecisionTreeClassifier()

# KNeighborsClassifier(18)
#SVC(kernel="rbf", C=1000, probability=True, gamma=0.0001),  # working
#DecisionTreeClassifier(),  # working
#RandomForestClassifier(n_estimators=100),  # working
#GradientBoostingClassifier(),  # working
#XGBClassifier()


p = predictor([transforms[t] for t in transforms], clf)
p.fit(X, y)

test_data = get_dataset('url-versions-2015-06-14-clean-test.csv')
X, y = split_data(test_data)

# example = X[X.articleId == '18eea9c0-47e3-11e4-9960-a165dd517bbb']
#
# p.predict_proba(example)

y_hat = p.predict(X)

Z = X.copy()
Z['y'] = y
Z['y_hat'] = y_hat

clean_test_errors = Z[(Z.y != Z.y_hat)][['claimHeadline', 'articleHeadline', 'y', 'y_hat', 'articleId', 'claimId']]
clean_test_errors[clean_test_errors.y_hat == 'against'].shape

clean_test_errors[clean_test_errors.articleId == '60669e60-c106-11e4-8747-5b024dd6dd6c']

Z[Z.y == Z.y_hat][['claimHeadline', 'articleHeadline', 'y', 'y_hat', 'articleId', 'claimId']].to_csv('../output/url-versions-2019-04-10-clean-test-correct_Xgb.csv')
Z[Z.y != Z.y_hat][['claimHeadline', 'articleHeadline', 'y', 'y_hat', 'articleId', 'claimId']].to_csv('../output/url-versions-2019-04-10-clean-test-errors_Xgb.csv')

accuracy_score(y, y_hat)
pd.Series(y).value_counts()

y.shape

pd.Series(y).value_counts() / pd.Series(y).value_counts().sum()