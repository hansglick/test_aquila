# Aquila test

### Solution au test
https://github.com/hansglick/test_aquila/blob/master/notebooks/Test_Technique_ML.ipynb

### Remarques
Le repo contient mes réponses au test Aquila. Je me suis servi des librairies suivantes pour les calculs : _pandas, numpy, scikit-learn_. Ayant quelques heures à tuer, **j'ai décidé d'implémenter une version du Kmodes clustering algorithm** afin de répondre à la question 8. Le module se trouve fun/fun.py. Il est importé à la cellule 22 du notebook. Le repo de cette implémentation se trouve à l'adresse suivante https://github.com/hansglick/kmodes :
 * fun.py : contient les fonctions de l'implémentation
 * test_kmodes_function.ipynb : un notebook qui teste l'implémentation sur des données simulées

### Durée du test
* Documentation sur le Kmodes + Implémentation : 4 heures
* Test Aquila : 2 heures 


### Instructions
 * Cloner le repo : `git clone https://github.com/hansglick/test_aquila.git`
 * Afin de reproduire l'environnement dans conda : `conda env create -f env/environment.yml`
 * Activer l'environnement : `source activate test_aquila`
