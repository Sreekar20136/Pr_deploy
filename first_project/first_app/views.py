from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.shortcuts import render,redirect
from .forms import StudentForm 
from .models import MLModel
from .Mailspamchecker import train_test_split,TfidfVectorizer,LogisticRegression,accuracy_score


def index(request) :
   fruits = ['apple', 'banana', 'kiwi', 'guava', 'mango']
   my_dict = { 'fruit_list': fruits }
   return render(request, 'index.html', my_dict)
        
def get_student(request):    
    if request.method == 'POST':          
      form = StudentForm(request.POST)     
      if form.is_valid():
          s_name = form.cleaned_data['name']
          s_roll = form.cleaned_data['roll']
          s_degree = form.cleaned_data['degree']        
          s_branch = form.cleaned_data['branch']
      return HttpResponseRedirect('student')
    else: 
        form =StudentForm()
        return render(request, 'studentForm.html', {'form': form})
  
def upload_dataset(request):
    if request.method == 'POST':
        dataset = request.FILES['dataset']
        ml_model = MLModel.objects.create(dataset=dataset)
        raw_mail_data = pd.read_csv(dataset)
        mail_data = raw_mail_data.where(pd.notnull(raw_mail_data),'')
        mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
        mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
        mail_data.head()
        X_Train,X_test,Y_Train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)
        feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
        X_train_feature = feature_extraction.fit_transform(X_Train)
        X_test_feature = feature_extraction.transform(X_test)
        Y_Train = Y_Train.astype('int')
        Y_test = Y_test.astype('int')
        print(X_train_feature)
        X_Train
# Training the Model
# Logistic Regression
        model = LogisticRegression()
        model.fit(X_train_feature,Y_Train)
# Evaluating the Trained Model
# Predition on Training Model
        prediction_on_Training_Data = model.predict(X_train_feature)
        accuracy_on_training_data = accuracy_score(Y_Train,prediction_on_Training_Data)
        print("Accuracy for Training : ",accuracy_on_training_data * 100)
        # Add code to process the dataset and calculate training accuracy
        # Update ml_model.training_accuracy with the calculated value
        return redirect('accuracy')  # Redirect to the accuracy view
    return render(request, 'upload.html')
def calculate_accuracy(request):
    ml_model = MLModel.objects.latest('id')  # Get the latest uploaded dataset
    # Add code to load the dataset, perform training, and calculate accuracy
    raw_mail_data = pd.read_csv(dataset)
    mail_data = raw_mail_data.where(pd.notnull(raw_mail_data),'')
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
    mail_data.head()
    X_Train,X_test,Y_Train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)
    feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
    X_train_feature = feature_extraction.fit_transform(X_Train)
    X_test_feature = feature_extraction.transform(X_test)
    Y_Train = Y_Train.astype('int')
    Y_test = Y_test.astype('int')
    print(X_train_feature)
    X_Train
# Training the Model
# Logistic Regression
    model = LogisticRegression()
    model.fit(X_train_feature,Y_Train)
# Evaluating the Trained Model
# Predition on Training Model
    prediction_on_Training_Data = model.predict(X_train_feature)
    calculated_accuracy = accuracy_score(Y_Train,prediction_on_Training_Data)
    print("Accuracy for Training : ",accuracy_on_training_data * 100)
    ml_model.training_accuracy = calculated_accuracy
    ml_model.save()
    return redirect('accuracy')  # Redirect to the accuracy view
