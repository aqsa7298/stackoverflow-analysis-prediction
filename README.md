# Stackoverflow Analysis and Prediction
## Project Overview
The purpose of this project is to get some useful insights about stackoverflow questions 2016-2020 and then perform predictive analysis. Different <b> natural language processing (NLP) techniques </b> like TFIDF, Doc2Vec etc. have been applied to extract relevant information. This information is then used by <b>machine learning models </b> to predict category of questions. The performace of each model is evaluated by using confusion matrix. The models used are as follow:
* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* Random Forest
* XGBoost
* Multilayer Perceptron (MLP)
## Dataset Description
The dataset is about stackoverflow questions from 2016-2020. There are two csv files with total 60000 enteries. Each question is classified into three categories:
* HQ: High-quality posts with a total of 30+ score and without a single edit.
* LQ_EDIT: Low-quality posts with a negative score, and multiple community edits. However, they still remain open after those changes.
* LQ_CLOSE: Low-quality posts that were closed by the community without a single edit. </br>
This dataset is available on Kaggle (link below) </br>
[stackoverflow 60k questions](https://www.kaggle.com/imoore/60k-stack-overflow-questions-with-quality-rate) 
## Dependencies
* Python 3.9 or above
* Anaconda 4.9 or above
## Installation and execution
1. Clone the project in desired directory.
2. Open anaconda prompt and go to the project directory.
3. Inside project directory run following command: 
</br>
<b>`conda env create --prefix ./myenv --file environment.yml`</b>
</br> 
it will create virtual environment myenv with all the required libraries.
</br>
4. Activate myenv by typing following command:
</br>
<b>`conda activate <complete path to project directory>\myenv` </b>
</br>
5. Run <b>`python main.py`</b>
</br>
It will first perform exploratory analysis and then predictive analysis of data.
</br>
Note: Each figure will be saved in images folder during program execution.
