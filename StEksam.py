#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import spacy_streamlit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso
import sklearn.metrics as sm
import spacy_streamlit
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection



def main():
    st.title("Eksamen Regression")

    st.header("Upload your CSV data file")
    data_file = st.file_uploader("Upload CSV", type=["csv"])

    if data_file is not None:
        data = pd.read_csv(data_file)
        if data is not None: 
            lst = ['health','income','inflation','Alcohol_consumption','Hepatitis_B','Measles','BMI','Polio','Diphtheria','Incidents_HIV','Schooling','Status','Birth_Rate','Infant_mortality','GDP','Gross_tertiary_education_enrollment','Life_expectancy','Maternal_mortality_ratio','Physicians_per_thousand','Unemployment_rate','Urban_population','fuel','GDP_per_capita']
            for i in lst:
                data[i].replace(0, np.nan, inplace=True)
                data[i].fillna(data[i].median(), inplace=True)
                
                
                
        
        st.header("Select Features and Target Variable")
        # Add a "Select All" option to the multi-select widget
        features = st.multiselect("Select Features", ["Select All"] + data.columns.tolist(), ["Select All"])
        target = st.selectbox("Select Target Variable", data.columns)

        # Check if "Select All" is selected, and if so, select all features
        if "Select All" in features:
            features = data.columns.tolist()
        

        st.write("Data overview:")
        st.write(data.head())

        X = data[features]  # Features
        y = data[target]  # Target variable
        #test
        lasso = Lasso(alpha = 1)
        

        st.header("Choose Regression Model")

        regression_model = st.selectbox("Select Regression Model", ["Multiple Linear Regression", "Lasso Regression"])

        st.header("Training the Model")

        if regression_model == "Multiple Linear Regression":
            model = LinearRegression()
        else:
            alpha = st.slider("Lasso Alpha (Regularization Strength)", 0.0, 100.0, 0.01)
            model = Lasso(alpha=alpha)
            

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)  # Calculate RMSE
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.header("Model Evaluation")
        st.write("Mean Squared Error (MSE):", mse)
        st.write("Root Mean Squared Error (RMSE):", rmse)
        st.write("Mean Absolute Error (MAE):", mae)
        st.write("R-squared (R2) Score:", r2)
        #test
        # calculate MAE using scikit-learn
        st.write("MAE: ", mean_absolute_error(y_test, y_pred))
        # calculate MSE using scikit-learn
        st.write("MSE: ", mean_squared_error(y_test, y_pred))
        # calculate RMSE using scikit-learn
        st.write("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
        st.write('R-squared (training) ', round(model.score(X_train, y_train), 3))
        st.write('R-squared (testing) ', round(model.score(X_test, y_test), 3))
        # R-squared
        #r2_score(y, predict(X))
        st.write("r2 score: ",r2_score(y_test, y_pred))
        st.write('Intercept: ', model.intercept_)
        st.write('Coefficient:', model.coef_)
        # Explained variance score: 1 is perfect prediction
        eV = round(sm.explained_variance_score(y_test, y_pred), 2)
        st.write('Explained variance score ',eV )
        st.write("test"), model.score(X_train, y_train)

        st.header("Make Predictions")

        st.write("Enter values for the selected features to make predictions:")
        input_features = {}
        for feature in features:
            input_features[feature] = st.number_input(f"Enter {feature}", value=0.0)
            
        if st.button("Make Prediction"):
            input_data = np.array([[input_features[feature] for feature in features]])
            prediction = model.predict(input_data)
            st.write(f"Predicted {target}: {prediction[0]}")

        #Når man opdatere resultatet
        st.header("Scatterplot of Actual vs. Predicted Values")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Scatterplot of Actual vs. Predicted Values")
        st.pyplot(fig)    
        
        # Create a heatmap of the correlation matrix
        #st.header("Heatmap of Feature Correlations")
        #correlation_matrix = X.corr()
        #fig, ax = plt.subplots(figsize=(40, 30))
        #sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        #st.pyplot(fig)

        
       
    
    
       
        
            
            
        
        
  
        
     
  



        

       
     

       


main()

