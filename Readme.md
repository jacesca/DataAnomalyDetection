# Machine Learning - Data Analysis Detection
Sample code to make data analysis detection.

Features:
- Point Anomalies (Outliers)
- Statistical methods:
    * Rule base approach (Heuristic method)
    * IQR rule
    * 3-sigma rule (68-95-99.7 rule or empirical rule)
    * MAD (Median Absolute Deviation) rule
- Machine Learning techniques:
    * KMeans
    * DBSCAN
- Regularization


## Run ML model
```
python SimpleOutliersDetection.py
python IqrOutliersDetection.py
python SigmaOutliersDetection.py
python MADRule.py
python KMenasOutlierDetection.py
python DBScanOutlierDetection.py
python regularization.py
```

## Installing using GitHub
- Fork the project into your GitHub
- Clone it into your dektop
```
git clone https://github.com/jacesca/DataAnomalyDetection.git
```
- Setup environment (it requires python3)
```
python -m venv venv
source venv/bin/activate  # for Unix-based system
venv\Scripts\activate  # for Windows
```
- Install requirements
```
pip install -r requirements.txt
```

## Others
- Proyect in GitHub: https://github.com/jacesca/DataAnomalyDetection
- Commands to save the environment requirements:
```
conda list -e > requirements.txt
# or
pip freeze > requirements.txt

conda env export > env.yml
```
- For coding style
```
black model.py
flake8 model.py
```

## Extra documentation
None
