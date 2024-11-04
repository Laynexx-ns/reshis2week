import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('data_list.csv')


def map_education_level(education_level):
    education_level_map = {
        "MASTER": 0.5,
        "BACHELOR": 0.3,
        "SPECIALITY": 0.1,
    }
    return education_level_map.get(education_level, 0)


data['education_level_numeric'] = data['education_level_code'].apply(map_education_level)


numeric_features = ['age', 'semester_cnt', 'subcide_rate', 'semester_cost_amt', 'initial_approved_amt', 'initial_term', 'education_level_numeric']

X = data[numeric_features]
y = data['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"R-squared на обучающей выборке: {train_score:.5f}")
print(f"R-squared на тестовой выборке: {test_score:.5f}")

for feature, coef in zip(numeric_features, model.coef_):
    print(f"{feature}: {coef:.5f}")

def predict_score(age, semester_cnt, subcide_rate, semester_cost_amt, initial_approved_amt, initial_term, education_level):
    dataw = {
        'age': [age],
        'semester_cnt': [semester_cnt],
        'subcide_rate': [subcide_rate],
        'semester_cost_amt': [semester_cost_amt],
        'initial_approved_amt': [initial_approved_amt],
        'initial_term': [initial_term],
        'education_level_numeric':  [map_education_level(education_level)]
    }
    df = pd.DataFrame(dataw)
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    return prediction[0]


# testing

age = data['age']
semester_cnt = data['semester_cnt']
subcide_rate = data['subcide_rate']
semester_cost_amt = data['semester_cost_amt']
initial_approved_amt = data['initial_approved_amt']
initial_term = data['initial_term']
education_level_code = data['education_level_code']

needed_score = data['score']



list_true = []
list_pred = []


for i in range(7976):

        print("_____________________________________________________")
        score = predict_score(age[i], semester_cnt[i], subcide_rate[i], semester_cost_amt[i], initial_approved_amt[i], initial_term[i], education_level_code[i])
        print((str(age[i]), str(semester_cnt[i]), str(subcide_rate[i]), str(semester_cost_amt[i]), str(initial_approved_amt[i]), str(initial_term[i]), str(education_level_code[i])))
        print()
        print(str(education_level_code[i]))
        print(map_education_level(education_level_code[i]))
        print("")
        print(f"Предсказанный score: {score:.4f}")
        list_pred.append(score)
        print(f"Необходимый score: {needed_score[i]}")
        list_true.append(needed_score[i])



def calculate_MAE(y_true, y_pred):
    mae_sklearn = mean_absolute_error(y_true, y_pred)
    print(f"MAE: {mae_sklearn}")
    return mae_sklearn

mae = calculate_MAE(list_true, list_pred)
print(mae)

