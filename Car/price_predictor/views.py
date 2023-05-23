from django.shortcuts import render
import requests
import pickle
import numpy as np
import sklearn

model = pickle.load(open('./models/random_forest_regression_model.pkl', 'rb'))

def index(request):
    return render(request,"index.html")

def prediction(request):
    if request.method== 'POST':

        Year=int(request.POST['Year'])
        Year = 2020 - Year

        Present_Price=float(request.POST['Present_Price'])

        Kms_Driven=int(request.POST['Kms_Driven'])
        Kms_Driven2 = np.log(Kms_Driven)

        Owner=int(request.POST['Owner'])

        Fuel_Type_Petrol=request.POST['Fuel_Type_Petrol']
        if (Fuel_Type_Petrol == 'Petrol'):
            Fuel_Type_Petrol = 1
            Fuel_Type_Diesel = 0
        else:
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 1

        Seller_Type_Individual=request.POST['Seller_Type_Individual']
        if (Seller_Type_Individual == 'Individual'):
            Seller_Type_Individual = 1
        else:
            Seller_Type_Individual = 0


        Transmission_Mannual=request.POST['Transmission_Mannual']
        if (Transmission_Mannual == 'Mannual'):
            Transmission_Mannual = 1
        else:
            Transmission_Mannual = 0

        prediction = model.predict([[Present_Price, Kms_Driven2, Owner, Year, Fuel_Type_Diesel, Fuel_Type_Petrol,
                                     Seller_Type_Individual, Transmission_Mannual]])

        output = round(prediction[0], 2)
        if output < 0:
            return render(request,'index.html',{'prediction_texts':"Sorry you cannot sell this car"})
        else:
            contex={"output":output}
            return render(request,'prediction.html',contex)
