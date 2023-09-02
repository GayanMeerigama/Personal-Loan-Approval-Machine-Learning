import pickle
import pandas
from sklearn.ensemble import RandomForestClassifier


df_Churn = pandas.read_csv("Churn_Modeling.csv")

# Random Forest:

model = RandomForestClassifier(random_state=20)


features = list(zip(df_Churn['CreditScore'],df_Churn['Age'],df_Churn['Tenure'],df_Churn['Balance'],df_Churn['NumOfProducts'],df_Churn['HasCrCard'],df_Churn['IsActiveMember'],df_Churn['EstimatedSalary']))


model.fit(features,df_Churn['Exited'])
pickle.dump(model, open('model.pkl','wb'))