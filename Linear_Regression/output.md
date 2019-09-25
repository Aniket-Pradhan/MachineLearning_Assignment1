## 1.1.a
### Linear Regression Gradient Descent for Abalone Dataset with k-fold cv, where k = 5
```
Training error for fold number: =  0 :  2.0585180660678764
Testing error for fold number: =  0 :  3.1605473220242053

Training error for fold number: =  1 :  2.4297372193525737
Testing error for fold number: =  1 :  1.30811009401006

Training error for fold number: =  2 :  2.2884699039383687
Testing error for fold number: =  2 :  2.450173831224953

Training error for fold number: =  3 :  2.384901645838306
Testing error for fold number: =  3 :  2.006047205462333

Training error for fold number: =  4 :  2.3648094460672584
Testing error for fold number: =  4 :  2.008517726912091

Average train error:  2.305287256252877
Average test error:  2.186679235926728
```

## 1.1.b
### Linear Regression Gradient Descent Normal Equation for Abalone Dataset with k-fold cv, where k = 5
```
Training error for fold number: =  0 :  1.9429775953591537
Testing error for fold number: =  0 :  3.5283513819355945

Training error for fold number: =  1 :  2.309082074895573
Testing error for fold number: =  1 :  1.654603780659608

Training error for fold number: =  2 :  2.1546109517250116
Testing error for fold number: =  2 :  2.4379772820413557

Training error for fold number: =  3 :  2.2579507835280186
Testing error for fold number: =  3 :  1.9551937922522502

Training error for fold number: =  4 :  2.2483955945684877
Testing error for fold number: =  4 :  1.9728616570660953

Average train error:  2.1826034000152488
Average test error:  2.309797578790981
```

## 1.2.a
### Linear Regression Gradient Descent with Ridge Regression for Abalone Dataset with k-fold cv, where k = 5
```
Finding the best hyperparameter for ridge regularization
Best hyperparameter for Ridge regularization:  0.7996554525892349
Training error for fold number: =  0 :  2.3335014402244303
Validation error for fold number: =  0 :  3.2623872224755344

Training error for fold number: =  1 :  2.5625566585540165
Validation error for fold number: =  1 :  2.4597996297202043

Training error for fold number: =  2 :  2.491450579601893
Validation error for fold number: =  2 :  2.818416208193973

Training error for fold number: =  3 :  2.5462710732527265
Validation error for fold number: =  3 :  2.4915289671128904

Training error for fold number: =  4 :  2.590348715060528
Validation error for fold number: =  4 :  2.100913620858779

Average train error:  2.504825693338719
Average validation error:  2.626609129672276
Test error: =  1.4973924574219633
```

## 1.2.b
### Linear Regression Gradient Descent with Lasso Regression for Abalone Dataset with k-fold cv, where k = 5
```
Finding the best hyperparameter for lasso regularization
Best hyperparameter for lasso regularization:  0.10280447320933092
Training error for fold number: =  0 :  2.2596076258834055
Validation error for fold number: =  0 :  3.0358884425269816

Training error for fold number: =  1 :  2.4775506106012104
Validation error for fold number: =  1 :  2.238702509644137

Training error for fold number: =  2 :  2.416044270291229
Validation error for fold number: =  2 :  2.593654946399076

Training error for fold number: =  3 :  2.4602355905740274
Validation error for fold number: =  3 :  2.271737881688197

Training error for fold number: =  4 :  2.5036126233401395
Validation error for fold number: =  4 :  1.8793048027042418

Average train error:  2.4234101441380025
Average test error:  2.4038577165925266
Test error: =  1.3272342068225447
```

## 1.3
### Linear Regression Gradient Descent using without regularization, with ridge, and with lasso
```
32.86766090878762 # error without regularization
33.096157721047646 # error for ridge
32.86766090878762 # error for lasso
```