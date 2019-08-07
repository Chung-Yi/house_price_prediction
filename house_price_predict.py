import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from scipy.stats import skew
from scipy.stats import skew, norm
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error


def model(train, y_train):
    lass = Lasso(alpha=0.1)
    bayes = BayesianRidge(
        n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06)
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    gbr = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        alpha=0.9)
    model_xgb = xgb.XGBRegressor(
        colsample_bytree=0.4603,
        gamma=0.0468,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=1.7817,
        n_estimators=2200,
        reg_alpha=0.4640,
        reg_lambda=0.8571,
        subsample=0.5213,
        silent=1,
        random_state=7,
        nthread=-1)

    # training model
    lass.fit(train, y_train)

    Lasso(
        alpha=0.1,
        copy_X=True,
        fit_intercept=True,
        max_iter=1000,
        normalize=False,
        positive=False,
        precompute=False,
        random_state=None,
        selection='cyclic',
        tol=0.0001,
        warm_start=False)
    bayes.fit(train, y_train)
    BayesianRidge(
        alpha_1=1e-06,
        alpha_2=1e-06,
        compute_score=False,
        copy_X=True,
        fit_intercept=True,
        lambda_1=1e-06,
        lambda_2=1e-06,
        n_iter=300,
        normalize=False,
        tol=0.001,
        verbose=False)
    regr.fit(train, y_train)
    RandomForestRegressor(
        bootstrap=True,
        criterion='mse',
        max_depth=2,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_split=1e-07,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=10,
        n_jobs=1,
        oob_score=False,
        random_state=0,
        verbose=0,
        warm_start=False)
    gbr.fit(train, y_train)
    GradientBoostingRegressor(
        alpha=0.9,
        criterion='friedman_mse',
        init=None,
        learning_rate=0.1,
        loss='ls',
        max_depth=3,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_split=1e-07,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=100,
        presort='auto',
        random_state=None,
        subsample=1.0,
        verbose=0,
        warm_start=False)
    model_xgb.fit(train, y_train)
    XGBRegressor(
        base_score=0.5,
        booster='gbtree',
        colsample_bylevel=1,
        colsample_bytree=0.4603,
        gamma=0.0468,
        learning_rate=0.05,
        max_delta_step=0,
        max_depth=3,
        min_child_weight=1.7817,
        missing=None,
        n_estimators=2200,
        n_jobs=1,
        nthread=-1,
        objective='reg:linear',
        random_state=7,
        reg_alpha=0.464,
        reg_lambda=0.8571,
        scale_pos_weight=1,
        seed=None,
        silent=1,
        subsample=0.5213)

    # result of prediction
    lass_predict = lass.predict(train)
    bayes_predict = bayes.predict(train)
    regr_predict = regr.predict(train)
    gbr_predict = gbr.predict(train)
    model_xgb_predict = model_xgb.predict(train)

    cv_score_lass = cross_val_score(
        lass, train, lass_predict, scoring='mean_squared_error', cv=5)
    cv_score_bayes = cross_val_score(
        bayes, train, bayes_predict, scoring='mean_squared_error', cv=5)
    cv_score_regr = cross_val_score(
        regr, train, regr_predict, scoring='mean_squared_error', cv=5)
    cv_score_gbr = cross_val_score(
        gbr, train, gbr_predict, scoring='mean_squared_error', cv=5)
    cv_score_model_xgb = cross_val_score(
        model_xgb,
        train,
        model_xgb_predict,
        scoring='mean_squared_error',
        cv=5)

    lass_score = np.min(cv_score_lass), np.max(cv_score_lass), np.std(
        cv_score_lass), np.mean(cv_score_lass)
    bayes_score = np.min(cv_score_bayes), np.max(cv_score_bayes), np.std(
        cv_score_bayes), np.mean(cv_score_bayes)
    regr_score = np.min(cv_score_regr), np.max(cv_score_regr), np.std(
        cv_score_regr), np.mean(cv_score_regr)
    gbr_score = np.min(cv_score_gbr), np.max(cv_score_gbr), np.std(
        cv_score_gbr), np.mean(cv_score_gbr)
    xgb_score = np.min(cv_score_model_xgb), np.max(cv_score_model_xgb), np.std(
        cv_score_model_xgb), np.mean(cv_score_model_xgb)

    total_score = pd.DataFrame({
        'la_score': lass_score,
        'bayes_score': bayes_score,
        're_score': regr_score,
        'gbr_score': gbr_score,
        'xgb_score': xgb_score
    },
                               index=['min', 'max', 'std', 'mean'])

    return total_score


def data_process():
    rcParams['figure.figsize'] = (12.0, 6.0)
    df_train = pd.read_csv('./data/train.csv')
    df_test = pd.read_csv('./data/test.csv')

    # describe data type (count, mean, std, min, 25%, 50%, 75%, max)
    print(f"numerical feature: {df_train.describe().shape}")
    print(df_train.describe())

    df_train['source'] = 'train'
    df_test['source'] = 'test'
    df_train.drop('building_id', axis=1, inplace=True)
    df_test.drop('building_id', axis=1, inplace=True)

    # kernel density plot
    sns.distplot(df_train.total_price, fit=norm)
    plt.ylabel('Frequency')
    plt.xlabel('total_price')
    (mu, sigma) = norm.fit(df_train['total_price'])
    fig = plt.figure()
    res = stats.probplot(df_train['total_price'], plot=plt)
    plt.show()
    print("skewness: %f" % df_train['total_price'].skew())
    print("kurtosis: %f" % df_train['total_price'].kurt())

    # log transform the target
    df_train['total_price'] = np.log1p(df_train['total_price'])

    # Kernel Density plot
    sns.distplot(df_train.total_price, fit=norm)
    plt.ylabel('Frequency')
    plt.title = ('SalePrice distribution')
    #Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(df_train['total_price'])
    # QQ plot
    fig = plt.figure()
    res = stats.probplot(df_train['total_price'], plot=plt)
    plt.show()

    # fig, ax = plt.subplots()
    # ax.scatter(x=df_train['parking_price'], y=df_train['total_price'])
    # plt.xlabel('parking_price')
    # plt.ylabel('total_price')
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.scatter(x=df_train['XIII_5000'], y=df_train['total_price'])
    # plt.xlabel('XIII_5000')
    # plt.ylabel('total_price')
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.scatter(x=df_train['XIII_10000'], y=df_train['total_price'])
    # plt.xlabel('XIII_10000')
    # plt.ylabel('total_price')
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.scatter(x=df_train['VII_10000'], y=df_train['total_price'])
    # plt.xlabel('VII_10000')
    # plt.ylabel('total_price')
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.scatter(x=df_train['IX_10000'], y=df_train['total_price'])
    # plt.xlabel('IX_10000')
    # plt.ylabel('total_price')
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.scatter(x=df_train['V_10000'], y=df_train['total_price'])
    # plt.xlabel('V_10000')
    # plt.ylabel('total_price')
    # plt.show()

    # # outlier deletion

    df_train = df_train.drop(
        df_train[(df_train['parking_price'] > 800000)].index)
    fig, ax = plt.subplots()
    ax.scatter(df_train['parking_price'], df_train['total_price'])
    plt.xlabel('parking_price')
    plt.ylabel('total_price')
    plt.show()

    # # combine data
    y_train = df_train['total_price']
    y_train = df_train.total_price.values
    total1 = pd.concat([df_train, df_test],
                       axis=0,
                       join='outer',
                       ignore_index=True)
    total1.drop(['total_price'], axis=1, inplace=True)
    # print(total1.shape)

    # # correration matrix
    # corrmat = df_train.corr()
    # f, ax = plt.subplots(figsize=(12, 9))
    # sns.heatmap(corrmat, vmax=0.9, square=True)
    # plt.show()

    # # get the top 10 more correlative features
    # cols = corrmat.nlargest(10, 'total_price')['total_price'].index
    # cm = np.corrcoef(df_train[cols].values.T)
    # plt.subplots(figsize=(12, 9))
    # sns.set(font_scale=1.25)
    # hm = sns.heatmap(
    #     cm,
    #     cbar=True,
    #     annot=True,
    #     square=True,
    #     fmt='.2f',
    #     annot_kws={'size': 10},
    #     yticklabels=cols.values,
    #     xticklabels=cols.values)
    # plt.yticks(rotation=0)
    # plt.xticks(rotation=90)
    # plt.show()

    # sns.set()
    # cols = [
    #     'total_price', 'parking_price', 'XIII_5000', 'jobschool_rate',
    #     'bachelor_rate', 'XIII_10000', 'VII_10000', 'IX_10000', 'V_10000',
    #     'master_rate'
    # ]
    # sns.pairplot(df_train[cols], size=1.25)
    # plt.show()

    # process missing data
    missing_data = total1.isnull().sum().sort_values(ascending=False)
    missing_precent = (
        (total1.isnull().sum()) / (total1.isnull().count())).sort_values(
            ascending=False)
    missing_type = total1.dtypes
    missing_all = pd.concat(
        [missing_data, missing_precent, missing_type],
        axis=1,
        keys=['missing_data', 'missing_precent', 'missing_type'])

    missing_all.drop(missing_all[missing_data == 0].index, inplace=True)
    missing_all.sort_values(by='missing_data', ascending=False)
    print(missing_all)

    total1.drop(missing_all[missing_data > 10000].index, axis=1, inplace=True)
    total1['village_income_median'] = total1['village_income_median'].fillna(
        total1['village_income_median'].mean())

    missing_data = total1.isnull().sum().sort_values(ascending=False)
    missing_precent = (
        (total1.isnull().sum()) / (total1.isnull().count())).sort_values(
            ascending=False)
    missing_type = total1.dtypes
    missing_all = pd.concat(
        [missing_data, missing_precent, missing_type],
        axis=1,
        keys=['missing_data', 'missing_precent', 'missing_type'])
    missing_all.sort_values(by='missing_data', ascending=False)
    print(missing_all)

    cols = total1.columns
    num_cols = total1._get_numeric_data().columns
    cate = list(set(cols) - set(num_cols))  # only building_id is category

    numer_feat = total1.dtypes[total1.dtypes != 'object'].index
    skewed_feat = total1[numer_feat].apply(lambda x: (x.dropna()).skew())
    skewed_feat = skewed_feat.sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feat})

    skewness = skewness[abs(skewness) > 0.75]
    skewness.dropna()
    skewness = skewness[abs(skewness) > 0.75]
    from scipy.special import boxcox1p
    skewness_feature = skewness.index
    lam = 0.15
    for i in skewness_feature:
        total1[i] = boxcox1p(total1[i], 0.15)
    print(skewness.head(10))

    # separate train and test data
    train = total1[total1['source'] == 'train']
    test = total1[total1['source'] == 'test']
    train.drop(['source'], axis=1, inplace=True)
    test.drop(['source'], axis=1, inplace=True)

    print('###########')
    missing_data = train.isnull().sum().sort_values(ascending=False)
    missing_precent = (
        (train.isnull().sum()) / (train.isnull().count())).sort_values(
            ascending=False)
    missing_type = train.dtypes
    missing_all = pd.concat(
        [missing_data, missing_precent, missing_type],
        axis=1,
        keys=['missing_data', 'missing_precent', 'missing_type'])
    missing_all.drop(missing_all[missing_data == 0].index, inplace=True)
    missing_all.sort_values(by='missing_data', ascending=False)

    print(missing_all)

    return (train, y_train)


def main():
    train, y_train = data_process()
    print(train.shape, y_train.shape)
    score = model(train, y_train)
    print(score)


if __name__ == "__main__":
    main()