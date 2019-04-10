import argparse
import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.join('..', 'src'))


from numpy import loadtxt
from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

from model.classifiers.lr_predictors import LogitPredictor, CompoundPredictor
# from model.classifiers.rf_predictors import RandomForestPredictor
from model.classifiers.classifier_showdown import ShowDownPredictor
from model.utils import get_dataset, split_data, RunCV, run_test

from model.baseline.transforms import (
    CosSimTransform,
    RefutingWordsTransform,
    QuestionMarkTransform,
    HedgingWordsTransform,
    InteractionTransform,
    NegationOfRefutingWordsTransform,
    BoWBTransform,
    BoUgTransform,
    BoBgTransform,
    PolarityTransform,
    BrownClusterPairTransform,
    BoWTransform
)

from model.ext.transforms import (
    AlignedPPDBSemanticTransform,
    NegationAlignmentTransform,
    Word2VecSimilaritySemanticTransform,
    DependencyRootDepthTransform,
    SVOTransform
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run_baseline cmd-line arguments.')

    # inc_transforms = [
    #     'Q',                # 1             1
    #     'BoW-Hed',          # 2-36         35
    #     'BoW-Ref',          # 37-48         12
    #     'I',                # 49-1272       1,224
    #     'BoW',              # 1273-1772     500
    #     'Sim-Algn-W2V',     # 1773-1773     1
    #     'Sim-Algn-PPDB',    # 1774-1774     1
    #     'Root-Dist',        # 1775-1776     2
    #     'Neg-Algn',         # 1777-1779     3
    #     'SVO',              # 1780-1788     9
    #     ]

    # default="Q,CosSim,BoUg,BoBg,BoW-B,PPDB,RootDep,NegAlgn,SVO,W2V",
    # Q,BoW,PPDB,RootDep,NegAlgn,SVO,W2V

    parser.add_argument('-f',
                        default="Q,BoW,PPDB,RootDep,NegAlgn,SVO,W2V",
                        type=str)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i', action='store_true')
    group.add_argument('-a', action='store_true')
    group.add_argument('-t', action='store_true')

    args = parser.parse_args()

    # When running original project, use LogitPredictor
    # predictor = LogitPredictor
    predictor = ShowDownPredictor

    train_data = get_dataset('url-versions-2015-06-14-clean-with-body-train-with-body.csv')
    X, y = split_data(train_data)
    # Split the dataset in two equal parts
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.5, random_state=0)

    test_data = get_dataset('url-versions-2015-06-14-clean-with-body-test-with-body.csv')

    transforms = {
        'CosSim': CosSimTransform,
        'BoW-B': BoWBTransform,
        'BoUg': BoUgTransform,
        'BoBg': BoBgTransform,
        'Q': QuestionMarkTransform,
        'W2V': Word2VecSimilaritySemanticTransform,
        'PPDB': AlignedPPDBSemanticTransform,
        'NegAlgn': NegationAlignmentTransform,
        'RootDep': DependencyRootDepthTransform,
        'SVO': SVOTransform,
        'BoW' : BoWBTransform,
    }

    inc_transforms = args.f.split(',')
    diff = set(inc_transforms).difference(transforms.keys())
    if diff:
        print 'Unrecognised features:', diff
        sys.exit(1)
    print 'Feature set:', inc_transforms
    if args.i:
        # incremental
        print 'Incremental test'
        df_out = pd.DataFrame(index=inc_transforms,
                              columns=['accuracy-cv', 'accuracy-test'], data=np.nan)
        inc_transforms_cls = []
        for i, k in enumerate(inc_transforms):
            inc_transforms_cls.append(transforms[k])
            print 'Using features:', inc_transforms[:i+1]

            p = predictor(inc_transforms_cls)
            cv_score = RunCV(X, y, p, display=True).run_cv()
            df_out.ix[k, 'accuracy-cv'] = cv_score.accuracy

            if args.t:
                # calculate test set accuracy score
                test_score = run_test(X, y, test_data, p, display=True)
                df_out.ix[k, 'accuracy-test'] = test_score.accuracy
        print(df_out)
    elif args.a:
        # ablation
        print 'Ablation test'

        if args.f:
            features = args.f
        else:
            features = "Q,CosSim,BoUg,BoBg,BoW-B,PPDB,RootDep,NegAlgn,SVO"
        ablations = [[x] for x in features.split(',')]
        df_out = pd.DataFrame(index=['-' + str(a) for a in ablations],
                              columns=['accuracy-cv', 'accuracy-test'], data=np.nan)
        inc_transforms_cls = [transforms[t] for t in inc_transforms]
        p = predictor(inc_transforms_cls)
        cv_score = RunCV(X, y, p, display=False).run_cv()
        print 'CV score: :{0:f}'.format(cv_score.accuracy)

        if args.t:
            test_score = run_test(X, y, test_data, p, display=False)
            print 'Test score: :{0:f}'.format(test_score.accuracy)

        for ablation in ablations:
            print 'Ablating: ' + str(ablation)
            inc_transforms_ablate = list(inc_transforms)
            for a in ablation:
                try:
                    inc_transforms_ablate.remove(a)
                except:
                    pass
            print 'Ablated feature set:' + str(inc_transforms_ablate)
            inc_transforms_cls_ablate = [transforms[t] for t in inc_transforms_ablate]
            p = predictor(inc_transforms_cls_ablate)
            cv_score_ablate = RunCV(X, y, p, display=False).run_cv()
            key = '-' + str(ablation)
            print 'Ablated CV score:', cv_score_ablate.accuracy
            df_out.ix[key, 'accuracy-cv'] = cv_score.accuracy - cv_score_ablate.accuracy

            if args.t:
                test_score_ablate = run_test(X, y, test_data, p, display=False)
                print 'Ablated test score:', test_score_ablate.accuracy
                df_out.ix[key, 'accuracy-test'] = test_score.accuracy - test_score_ablate.accuracy

        print(df_out * 100.0)
    else:
        classifiers = [
            KNeighborsClassifier(18),  # working
            #SVC(kernel="rbf", C=1000, probability=True, gamma=0.0001),  # working
            #DecisionTreeClassifier(),  # working
            #RandomForestClassifier(n_estimators=100),  # working
            #GradientBoostingClassifier(),  # working
            #XGBClassifier()
        ]

         # p = predictor([transforms[t] for t in inc_transforms])
         # cv_score = RunCV(X, y, p, display=True).run_cv()
         # print 'CV score: ', cv_score.accuracy
         # if args.t:
         #    print 'arg f is set: ', args.t
         #    test_score = run_test(X, y, test_data, p, display=True)

        # for clf in classifiers:
        #     p = predictor([transforms[t] for t in inc_transforms], clf)
        #     print clf
        #     cv_score = RunCV(X, y, p, display=True).run_cv()




