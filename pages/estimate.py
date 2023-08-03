import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout='wide',page_title="Regression Automation")
st.title("Regression Automation ")
def upload_csv():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df

data=pd.DataFrame(upload_csv())
col= st.columns([1,6])
with col[0]:
    if st.button("Show Table"):
        with col[1]:
            st.table(data.head(10))

with col[0]:
    if st.button("Description"):
        with col[1]:
            st.table(data.describe())






#####
features= st.multiselect("Select columns to include in data",data.columns)
data= data[features]
st.table(data.head(5))



if st.button("show missing values"):
    st.table(data.isna().sum())








#st.table(data.dtypes)


#Test for removing missing values
"# select target variable"
target= st.selectbox("select target feature",data.columns)

if (target) is not None and data[target].dtype in ['int64', 'float64'] :
    X= data.drop([target], axis=1)
    Y=data[target]

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test= train_test_split(X,Y,test_size=0.2, random_state=0)




    "# Select algorithm"
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    categorical= [x for x in X.columns if X[x].dtype not in ['float64', 'int64']]

    cf= ColumnTransformer([('trf',OneHotEncoder(sparse=False,drop='first'), 
    categorical)],remainder='passthrough')

    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import r2_score, mean_absolute_error

    algo= ['Gradient','Linear regression', "KNN",'Decision tree','Random forest', 'SVR']
    algo_select= st.selectbox('select algo', algo)
    if algo_select=="Gradient":
        pipe= Pipeline(steps=[
            ('step1',cf),
            ('step2', GradientBoostingRegressor())
        ])

        #fit data
        pipe.fit(x_train,y_train)
        y_pred= pipe.predict(x_test)
        st.write(f"R-squared:{r2_score( y_test, y_pred)}")
        st.write("Mean absolute error: ",mean_absolute_error(y_test, y_pred))
        predict={}
        for i in x_train.columns:
            if x_train[i].dtype in ['int64','float64']:
                q= st.number_input(f"Enter value for {i}")
                predict[i]=[q]
            else:
                p= st.selectbox(f'choose value for {i}', x_train[i].unique())
                predict[i]=[p]
        
        st.write(pipe.predict(pd.DataFrame(predict)))

        #Linear regresion
    elif algo_select=="Linear regression":
        pipe= Pipeline(steps=[
            ('step1',cf),
            ('step2', LinearRegression())
        ])

        #fit data
        pipe.fit(x_train,y_train)
        y_pred= pipe.predict(x_test)
        st.write(f"R-squared:{r2_score(y_test, y_pred)}")
        st.write("Mean absolute error: ",mean_absolute_error(y_test, y_pred))
        predict={}
        for i in x_train.columns:
            if x_train[i].dtype in ['int64','float64']:
                q= st.number_input(f"Enter value for {i}")
                predict[i]=[q]
            else:
                p= st.selectbox(f'choose value for {i}', x_train[i].unique())
                predict[i]=[p]
        
        st.write(pipe.predict(pd.DataFrame(predict)))

    #for KNN
    elif algo_select=="KNN":
        pipe= Pipeline(steps=[
            ('step1',cf),
            ('step2', KNeighborsRegressor(n_neighbors=5))
        ])

        #fit data
        pipe.fit(x_train,y_train)
        y_pred= pipe.predict(x_test)
        st.write(f"R-squared:{r2_score(y_test, y_pred)}")
        st.write("Mean absolute error: ",mean_absolute_error(y_test, y_pred))
        predict={}
        for i in x_train.columns:
            if x_train[i].dtype in ['int64','float64']:
                q= st.number_input(f"Enter value for {i}")
                predict[i]=[q]
            else:
                p= st.selectbox(f'choose value for {i}', x_train[i].unique())
                predict[i]=[p]
        
        st.write(pipe.predict(pd.DataFrame(predict)))

    #Decision tree
    elif algo_select=="Decision tree":
        pipe= Pipeline(steps=[
            ('step1',cf),
            ('step2', DecisionTreeRegressor())
        ])

        #fit data
        pipe.fit(x_train,y_train)
        y_pred= pipe.predict(x_test)
        st.write(f"R-squared:{r2_score(y_test, y_pred)}")
        st.write("Mean absolute error: ",mean_absolute_error(y_test, y_pred))
        predict={}
        for i in x_train.columns:
            if x_train[i].dtype in ['int64','float64']:
                q= st.number_input(f"Enter value for {i}")
                predict[i]=[q]
            else:
                p= st.selectbox(f'choose value for {i}', x_train[i].unique())
                predict[i]=[p]
        
        st.write(pipe.predict(pd.DataFrame(predict)))

    # Random forest
    elif algo_select=="Random forest":
        pipe= Pipeline(steps=[
            ('step1',cf),
            ('step2', RandomForestRegressor())
        ])

        #fit data
        pipe.fit(x_train,y_train)
        y_pred= pipe.predict(x_test)
        st.write(f"R-squared:{r2_score(y_test, y_pred)}")
        st.write("Mean absolute error: ",mean_absolute_error(y_test, y_pred))
        predict={}
        for i in x_train.columns:
            if x_train[i].dtype in ['int64','float64']:
                q= st.number_input(f"Enter value for {i}")
                predict[i]=[q]
            else:
                p= st.selectbox(f'choose value for {i}', x_train[i].unique())
                predict[i]=[p]
        
        st.write(pipe.predict(pd.DataFrame(predict)))

    #SVR
    elif algo_select=="SVR":
        pipe= Pipeline(steps=[
            ('step1',cf),
            ('step2', SVR())
        ])

        #fit data
        pipe.fit(x_train,y_train)
        y_pred= pipe.predict(x_test)
        st.write(f"R-squared:{r2_score(y_test, y_pred)}")
        st.write("Mean absolute error: ",mean_absolute_error(y_test, y_pred))
        predict={}
        for i in x_train.columns:
            if x_train[i].dtype in ['int64','float64']:
                q= st.number_input(f"Enter value for {i}")
                predict[i]=[q]
            else:
                p= st.selectbox(f'choose value for {i}', x_train[i].unique())
                predict[i]=[p]
        
        st.write(pipe.predict(pd.DataFrame(predict)))
