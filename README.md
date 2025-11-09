Please read the Apziva___Project_A__Happy_Customers.pdf for comprehensive results of this project.

In Project A - Customer Happiness\Main.py, there are three main sections of my model.

1. def stacking_accuracy_optimization_deterministic(X, y, test_size=0.3, target_accuracy=0.65) from line 39 to 373， which basically outlines my stacking model (LR, RF, SVM);
2. def rfe_analysis_existing_model(X, y, existing_model, model_name="Existing Model") from line 380 to 536, which uses RFE to measure the importance of features and their relations;
3. def hyperopt_stacking_optimization(X, y, test_size=0.3, n_evals=50) from line 546 to the end, which tune the hyper parameters to optimize the model.
