###Classification
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


st.set_page_config(layout='wide',page_title="Regression Automation")
st.title("Logistic Regression/Classification Automation ")

def upload_csv():
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df

st.sidebar.title("Hey there.")
data=pd.DataFrame(upload_csv())

if st.sidebar.button("Show Table"):
    
    st.table(data.head(10))

if st.sidebar.button("Description"):
    
    st.table(data.describe())

features= st.multiselect("Select columns to include in data",data.columns)
data= data[features]
st.table(data.head(5))



if st.sidebar.button("show missing values"):
    
    st.table(data.isna().sum())

target= st.selectbox("select target feature",data.columns)




def regressor(X,Y):
    #from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test= train_test_split(X,Y,test_size=0.2, random_state=0)



    "# Select algorithm"
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    categorical= [x for x in X.columns if X[x].dtype not in ['float64', 'int64']]

    cf= ColumnTransformer([('trf',OneHotEncoder(sparse=False,drop='first'), 
    categorical)],remainder='passthrough')

    from sklearn.preprocessing import StandardScaler
    sc= StandardScaler()

    
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score

    
    algo= ['Gradient','Logistic regression', "KNN",'Decision tree','Random forest', 'SVC']
    algo_select= st.selectbox('select algo', algo)
    if algo_select=="Gradient":
        pipe= Pipeline(steps=[
            ('step1',cf),
            ('Step2',sc),
            ('step3', GradientBoostingClassifier())
        ])

        #fit data
        pipe.fit(x_train,y_train)
        y_pred= pipe.predict(x_test)
        st.write(f"Accuracy score :{accuracy_score( y_test, y_pred)}")
        st.write("Precision score: ",precision_score(y_test, y_pred))
        predict={}
        for i in x_train.columns:
            if x_train[i].dtype in ['int64','float64']:
                q= st.number_input(f"Enter value for {i}", value=0, step=1)
                predict[i]=[q]
            else:
                p= st.selectbox(f'choose value for {i}', x_train[i].unique())
                predict[i]=[p]
        
        st.write(pipe.predict(pd.DataFrame(predict)))

        ##Logistic Regression
        
    elif algo_select=="Logistic regression":
        pipe= Pipeline(steps=[
            ('step1',cf),
            ('Step2',sc),
            ('step3', LogisticRegression())
        ])

        #fit data
        pipe.fit(x_train,y_train)
        y_pred= pipe.predict(x_test)
        st.write(f"Accuracy score :{accuracy_score( y_test, y_pred)}")
        st.write("Precision score: ",precision_score(y_test, y_pred))
        predict={}
        for i in x_train.columns:
            if x_train[i].dtype in ['int64','float64']:
                q= st.number_input(f"Enter value for {i}", value=0, step=1)
                predict[i]=[q]
            else:
                p= st.selectbox(f'choose value for {i}', x_train[i].unique())
                predict[i]=[p]
        
        st.write(pipe.predict(pd.DataFrame(predict)))

        #KNN Classifier
    
    elif algo_select=="KNN":
        pipe= Pipeline(steps=[
            ('step1',cf),
            ('Step2',sc),
            ('step3', KNeighborsClassifier())
        ])

        #fit data
        pipe.fit(x_train,y_train)
        y_pred= pipe.predict(x_test)
        st.write(f"Accuracy score :{accuracy_score( y_test, y_pred)}")
        st.write("Precision score: ",precision_score(y_test, y_pred))
        predict={}
        for i in x_train.columns:
            if x_train[i].dtype in ['int64','float64']:
                q= st.number_input(f"Enter value for {i}", value=0, step=1)
                predict[i]=[q]
            else:
                p= st.selectbox(f'choose value for {i}', x_train[i].unique())
                predict[i]=[p]
        
        st.write(pipe.predict(pd.DataFrame(predict)))

        #Decision Tree Regressor
    elif algo_select=="Decision tree":
        pipe= Pipeline(steps=[
            ('step1',cf),
            ('Step2',sc),
            ('step3', DecisionTreeClassifier())
        ])

        #fit data
        pipe.fit(x_train,y_train)
        y_pred= pipe.predict(x_test)
        st.write(f"Accuracy score :{accuracy_score( y_test, y_pred)}")
        st.write("Precision score: ",precision_score(y_test, y_pred))
        predict={}
        for i in x_train.columns:
            if x_train[i].dtype in ['int64','float64']:
                q= st.number_input(f"Enter value for {i}", value=0, step=1)
                predict[i]=[q]
            else:
                p= st.selectbox(f'choose value for {i}', x_train[i].unique())
                predict[i]=[p]
        
        st.write(pipe.predict(pd.DataFrame(predict)))

        #Random forest
    elif algo_select=="Random forest":
        pipe= Pipeline(steps=[
            ('step1',cf),
            ('Step2',sc),
            ('step3', RandomForestClassifier())
        ])

        #fit data
        pipe.fit(x_train,y_train)
        y_pred= pipe.predict(x_test)
        st.write(f"Accuracy score :{accuracy_score( y_test, y_pred)}")
        st.write("Precision score: ",precision_score(y_test, y_pred))
        predict={}
        for i in x_train.columns:
            if x_train[i].dtype in ['int64','float64']:
                q= st.number_input(f"Enter value for {i}", value=0, step=1)
                predict[i]=[q]
            else:
                p= st.selectbox(f'choose value for {i}', x_train[i].unique())
                predict[i]=[p]
        
        st.write(pipe.predict(pd.DataFrame(predict)))

        #SVC
    elif algo_select=="SVC":
        pipe= Pipeline(steps=[
            ('step1',cf),
            ('Step2',sc),
            ('step3', SVC())
        ])

        #fit data
        pipe.fit(x_train,y_train)
        y_pred= pipe.predict(x_test)
        st.write(f"Accuracy score :{accuracy_score( y_test, y_pred)}")
        st.write("Precision score: ",precision_score(y_test, y_pred))
        predict={}
        for i in x_train.columns:
            if x_train[i].dtype in ['int64','float64']:
                q= st.number_input(f"Enter value for {i}", value=0, step=1)
                predict[i]=[q]
            else:
                p= st.selectbox(f'choose value for {i}', x_train[i].unique())
                predict[i]=[p]
        
        st.write(pipe.predict(pd.DataFrame(predict)))
    
    
    





if (target) is not None and data[target].dtype in ['int64', 'float64'] :
    
    X= data.drop([target], axis=1)
    Y=data[target]
    regressor(X,Y)

    
elif (target) is not None and data[target].dtype not in ['int64', 'float64'] :
    unique= data[target].unique()
    data[target]= data[target].map({unique[0]:0,unique[1]:1})
    st.text({unique[0]:0,unique[1]:1})
    X= data.drop([target], axis=1)
    Y=data[target]
    regressor(X,Y)


        

