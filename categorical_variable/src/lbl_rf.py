# Label Encoder
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    df = pd.read_csv("../input/train_folds.csv")
    features = [f for f in df.columns if f not in ('id', 'target', 'kfold')]

    # fill all the NaN values with NONE
    # converting all the columns to string because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # Now it's time to label encode the features
    for col in features:
        lbl = preprocessing.LabelEncoder()
        # fit the label encoder for all the data
        lbl.fit(df[col])
        # transform all the data
        df.loc[:, col] = lbl.transform(df[col])




    # get train and validation data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training and validation data
    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # initialize random forest model
    model = ensemble.RandomForestClassifier(n_jobs=-1)

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
