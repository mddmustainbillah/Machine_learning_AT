import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    df = pd.read_csv("../input/train_folds.csv")
    features = [f for f in df.columns if f not in ('id', 'target', 'kfold')]

    # fill all the NaN values with NONE
    # converting all the columns to string because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # get train and validation data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initialize OneHotEncoder
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training and validation feature
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data[features])

    # transform training and validation data
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    # initialize logistic regression model
    model = linear_model.LogisticRegression()

    # fit the model on the training data (ohe)
    model.fit(x_train, df_train.target.values)

    # predict on validation data
    # we need to probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc acu score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"Fold: {fold}, AUC: {auc}")


if __name__ == "__main__":
    for i in range(5):
        run(i)
