# IEEE_CIS Fraud Detection - ML Assignment 2

## კონკურსის მიმოხილვა

ეს კონკურსი ეხება კლასიფიკაციის ამოცანას, კერძოდ ჩვენი მიზანია მოცემული ტრანზაქციის მიხედვით გავიგოთ ეს ტრანზაქცია ნამდვილია თუ ყალბი მისი მახასიათებლების საფუძველზე.
ამოცანა ფასდება AUROC მეტრიკით (Area under the ROC curve) predicted ალბათობასა და observed target-ს შორის.

## მიდგომა
ეს ამოცანა კლასიფიკაციის ამოცანაა, ვინაიდან, უნდა გავარკვიოთ ტრანზაქცია ნამდვილია თუ არა (ანუ გვაქვს ორი target მნიშვნელობა). შესაბამისად უნდა გამოვიყენოთ კლასიფიკაციის მოდელები. ჩემ მიერ არჩეული მოდელები შემდეგია: **Logistic Regression**, **Random Forest**, **AdaBoost** და **XGBoost**. აქედან ერთი წრფივი მოდელია. დანარჩენები კი ensemble მოდელებია. თუმცა სანამ მათ ტრენინგზე გადავიდოდი შევხედე მონაცემებს EDA ფაზაში(ამის ნოუთბუქიც ცალკე მაქვს ყველა სხვა ნოთბუქში ცალცალკე რომ აღარ მეწერა). ყველა მოდელისთვის data-ს ვყოფ 80/20 ად, trainging და test სეტად. თითოეული მოდელისთვის შემდეგი ფაზაა **Feature Cleaning / Preprocessing** ფაზა სადაც ისეთ სვეტებს ვყრი, რომელსაც ბევრი NAN მნიშვნელობა აქვს. შემდეგი ფაზაა Feature Engineering სადაც ახალ feature-ებს ვამატებ. ასევე კატეგორიულ ცვლადებს გარდავქმნი რიცხვითში, რათა მოდელებმა იმუშაონ და დარჩენილ სვეტებს missing value-თი ვავსებ Imputer ით. შემდეგ მოდის **Feature Selection** ფაზა სადაც თითოეული მოდელისთვის სხვადასვა selection მეთოდებს ვტესტავ, ვირჩევ საუკეთესოს და საბოლოო train და test სეტით ვატრენინგებ მოდელებს. საბოლოოდ ყველა run-ს ვლოგავ mlflow -ზე და თითოეულისთვის საუკეთესო pipeline-ს ვლოგავ მისი მოდელით, რადგან inference ში შემეძლოს raw სეტზე პაიპლაინის პირდაპირ გაშვება.

## რეპოზიტორიის სტრუქტურა

```
ML_assignment_2/
├── EDA.ipynb   
├── model_experiment_LogisticRegression.ipynb   
├── model_experiment_RandomForest.ipynb     
├── model_experiment_AdaBoost.ipynb    
├── model_experiment_XGBoost.ipynb     
├── model-inference.ipynb
└── README.md
```
## ფაილების აღწერა

| ფაილი | აღწერა |
|-------|--------|
| `EDA.ipynb ` | ნოუთბუქი Explanatory Data Analysis -თვის |
| `model_experiment_LogisticRegression.ipynb ` | ნოუთბუქი LogisticRegression-ის ტრენინგისთივის |
| `model_experiment_RandomForest.ipynb  ` | ნოუთბუქი RandomForest-ის ტრენინგისთივის |
| `model_experiment_AdaBoost.ipynb ` | ნოუთბუქი AdaBoost-ის ტრენინგისთივის |
| `model_experiment_XGBoost.ipynb    ` | ნოუთბუქი XGBoost-ის ტრენინგისთივის |
| `model-inference.ipynb` | საუკეთესო მოდელის ჩამოტვირთვა და kaggle-სთვის submission-ის შექმნა |
| `README.md` | პროექტის დოკუმენტაცია |

---

## მონაცემთა დამუშავება / გაწმენდა (Data Preprocessing / Cleaning)

### 1. Useless სვეტების გადაყრა

`TransactionID` არის უბრალო ID-ი, რომელსაც არანაირი predictive ღირებულება არ აქვს და მოდელის შესწავლისას მხოლოდ noise-ი იქნება, რადგან ის უბრალოდ ტრანზაქციის უნიკალური იდენთიფიკატორია(ეს კი შეიძლება შემთხვევითად გენერირებულიც არის). შესაბამისად ის გადავყარე.

### 2. NaN მნიშვნელობებიანი სვეტების დამუშავება

დატასეტის ერთ-ერთი მთავარი გამოწვევაა missing values ანუ იგვე NAN მნიშვნელობები. შესაბამისად შემოვიღე ესეთი მეთოდი. მაქვს ორი threshold-ი:

- **>75% NaN** - მთლიანად ვდროპავ სვეტს.
- **25%-დან 75%-მდე NaN** - ვამატებ binary missing flag-ს `<სვეტის სახელი>_missing` სახელით. NaN-ის ყოფნა შეიძლება ინფორმაციული სიგნალი იყოს ამიტომ, ვარჩიე რომ თუ ძალიან ბევრი missing არ აქვს სვეტს მაშინ უბრალოდ ეს სვეტები მომენიშნა.

### 3. Imputation

დარჩენილი NaN-ებისთვის ვიყენებ `SimpleImputer(strategy='median')`-ს, რადგან median უფრო მდგრადია outlier-ების მიმართ ვიდრე mean.

- **XGBoost-ში Imputation:** XGBoost-ს არ სჭირდება missing value ების შევსება Imputer-ით. ყოველ split-ზე ის თვითონ წყვეტს, რომელ მხარეს გაგზავნოს missing values. ამიტომ XGBoost-ის pipeline-ში imputer-ს არ ვიყენებ. ეს ერთ-ერთი მისი მთავარი უპირატესობაა სხვა მოდელებთან შედარებით.

---

## Feature Selection

- ეს ფაზა XGBoost-ს არ ეხება, რადგან მას ჩაშენებული selector-ი ისედაც აქვს.

- Logistic Regression-ისთვის გავტესტე ორი სხვადასხვა(Correlation only და IV + Correlation) ხოლო tree-based მოდელებისთვის (RF, AdaBoost) დავტესტე სამი სხვადასხვა სელექციის/ფილტრის სტრატეგია(Correlation only, IV + Correlation, TreeImportance):


### 1. Correlation Filter

ფილტრავს იმ ცვლადებს, რომლებიც ერთმანეთთან 0.9-ზე მეტადაა კორელირებული. (0.9 threshold ი ისედაც ბევრ ცვლადს ყრიდა ამიტომ მასზე დაბალი აღარ ავიღე). წყვილებიდან ერთს ტოვებს ეს ფილტრი (ნაკლებად redundant-ს). სვეტებს შორის საბოლოოდ კოლინეარულობა მცირდება.  სვეტების რაოდენობა 277-დან **148**-მდე მცირდება.

### 2. IV (Information Value) + Correlation

IV (Information Value) ზომავს თითოეული feature-ის predictive power-ს. threshold = 0.02  IV თუ ნაკლებია 0.02 ზე ამ სვეტებს გადავყრით). შემდეგ Correlation Filter-ი ფილტრავს redundant ცვლადებს. სვეტების რაოდენობა 277 -> 192 -> **106**.

### 3. Tree Importance

ვატრენინგებ პატარა Random Forest-ს (n_estimators=50, max_depth=10) პარამეტრებით,  ვტოვებ top 80 ცვლადს რომელსაც ყველაზე დიდი importance მიანიჭა ამ RandomForest-მა. ამ მიდგომამ **tree-based** მოდელებში საუკეთესო შედეგი მოიტანა, ვინაიდან tree importance იჭერს არაწრფივ დამოკიდებულებებს, რომელსაც ზედა ორი ფილტრი შეიძლება ვერ ამჩნევდეს კარგად.

--
ეს სამივე ფილტრი გავტესტე baseline მოდელებზე ერთჯერადად. საუკეთესო შედეგის მქონე filter თითოეული მოდელისთვის ავირჩიე საბოლოო ფილტრად Training და Hyperparameter tuning ფაზისთვის. LR-ისთვის IV+Corr ავირჩიე იმიტომ, რომ 30% ნაკლები feature-ით (106 vs 148) იგივე შედეგი მიიღო დაახლოებით ამიტომ, ტრენინგი უფრო სწრაფი იქნებოდა tuning-ის ფაზაში. Tree based მოდელებში კი TreeImportance ფილტრმა უკეთესი შედეგი დადო, ვიდრე ამ ფილტრის გარეშე.

(ფილტრს იგივე Feature Selector-ს ვეძახი) 

| მოდელი | Correlation only | IV+Corr | Tree Importance  | საბოლოო არჩევანი |
|--------|------------------------|---------------|----------------------|--------------|
| **LR** | val AUC: 0.8296 | val AUC: 0.8291 | — | **IV+Corr** (ნაკლები features და თითქმის იგივე AUC) |
| **RF** | val AUC: 0.8855 | val AUC: 0.8881 | val AUC: 0.8986 | **Tree Importance** |
| **AdaBoost** | val AUC: 0.8440 | val AUC: 0.8418 | val AUC: 0.8589 | **Tree Importance** |


---
## ტრენინგი და ექსპერიმენტები

| მეტრიკა | აღწერა |
|---------|--------|
| `train_auc` | AUC train set-ზე |
| `test_auc` | AUC test set-ზე |
| `auc_diff` | train - test (overfitting indicator) |
| `train_f1`, `test_f1` | F1 score train/test |
| `train_precision`, `test_precision` | Precision train/test |
| `train_recall`, `test_recall` | Recall train/test |

class imbalance-ის გასათვალისწინებლად ვიყენებ `class_weight='balanced'. როგორც EDA ში ვნახეთ დაახლოებით 97% ია non-fraud ტრანზაქცია.


### 1. Logistic Regression

ვტესტავ სხვადასხვა `C`-ს (C რეგულარიზაციის პარამეტრია) და `penalty` (l1/l2). LR-ისთვის ცალკე Cross Validation (5-fold) გავუშვი default config-ზე. საშუალო val AUC = 0.8283 ± 0.0031, რამაც მიმანიშნა, რომ ეს სტაბილური მოდელია და აქამდე ყველაფერი ნორმალურად გავაკეთე.

#### Logistic Regression შედეგები (top 3 best + top 3 worst)

**Top 3 საუკეთესო მოდელი test_auc მიხედვით:**

| hyperparams | train_auc | test_auc | diff | test_f1 | test_recall | test_precision |
|--------|-----------|----------|------|---------|-------------|----------------|
| **C=0.1, l1** | 0.8298 | **0.8291** | +0.0007 | 0.1735 | 0.7464 | 0.0981 |
| C=1.0, l1 | 0.8298 | 0.8291 | +0.0007 | 0.1735 | 0.7464 | 0.0981 |
| C=10.0, l1 | 0.8298 | 0.8291 | +0.0007 | 0.1735 | 0.7464 | 0.0981 |


