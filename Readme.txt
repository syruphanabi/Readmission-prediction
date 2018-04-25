The source data is ADMISSIONS.csv and NOTEEVENTS.csv



Part1: identify readmission case
	
	code:
		readmission.R

	Input:
		ADMISSIONS.csv

	Output:
		readmission.csv

	run:
		In R console, source("readmission.R")
		


Part2: join readmission labels to "discharge summary" notes, get (label, TEXT) pair.
	
	code: 
		cleaning.py

	Input:
		readmission.csv
		NOTEEVENTS.csv

	Output:
		p_set.csv
		n_set.csv



# with OpenRefine, eliminate all "\n" in TEXT, then
Part3: eliminate all non-A-to-Z and non-a-to-z characters

	code: 
		data_processing.json

	Input:
		p_set_m.csv
		n_set_m.csv

	Output:
		one_set_processed.csv
		zero_set_processed.csv

	run:
		this is a zeppelin notebook file that runs on Spark interpreter.



Part4: feature extraction with TF-IDF or word2vec. Combine them.
	
	code:
		feature-TF-IDF.py
		feature-word2vec.py
		feature-TFIDF-word-combine.py

	Input:
		data/one_set_processed.csv
		data/zero_set_processed.csv

	Output:
		Combined/Uni/word2vec2000.train/test/whole in feature folder
		


Part5: train prediction model and get metrics

	code:
		main.py
		parameter_tuning.py
		run_models.py
		ffnet_model.py

	Input:
		Combined/Uni/word2vec2000.train/test/whole in feature folder

	Output:
		figure

	note:
		main.py call the other 3 files.
		parameter_tuning.py is to select best params for Logistic Regression, SVM and Random Forest with 10-fold.
		run_models.py is to run Logistic Regression, SVM and Random Forest with best parameters.
		ffnet_model.py is run feed forward net with given parameters, and it can run seperately.



Part6: utils.py is shared by multiple .py files. Used to dump svmlight files, calculate metrics, draw ROC curves, etc.



