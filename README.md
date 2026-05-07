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