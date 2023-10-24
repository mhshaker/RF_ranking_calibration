import Data.data_provider as dp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from estimators.IR_RF_estimator import IR_RF


data_list = [
                  "cm1",
                  "datatrieve",
                  "kc1_class_level_defectiveornot",
                  "kc1",
                  "kc2",
                  "kc3",
                  "pc1",
                  "spect",
                  "spectf",
                  "vertebral", 
                  "wilt",
                  "parkinsons", 
                  "heart",
                  "wdbc",
                  "bank", 
                  "ionosphere", 
                  "HRCompetencyScores",
                  "spambase", 
                  "QSAR", 
                  "diabetes", 
                  "breast", 
                  "SPF",
                  "hillvalley",
                  "pc4",
                  "scene",
                  "Sonar_Mine_Rock_Data",
                  "Customer_Churn",
                  "jm1",
                  "eeg",
                  "phoneme"
            ]

for data in data_list:
    X, y = dp.load_data(data, "./")

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)

    clf = IR_RF(oob_score=False, random_state=0)
    clf.fit(x_train, y_train)

    clf_oob = IR_RF(oob_score=True, random_state=0)
    clf_oob.fit(x_train, y_train)

    if clf.score(x_test, y_test) != clf_oob.score(x_test, y_test):
        print(f"{data} error")
    else:
        print("fine")