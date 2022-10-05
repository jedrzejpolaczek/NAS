# General approach
Algorithm should have structure like this:
1. Encode data
2. Evaluate data
3. Feed GA with data
4. Exploit data 
* NOTE: repeate from step 1 till... how long we want tbh.

# I am working on...
1. NGSA-Net - https://github.com/ianwhale/nsga-net

## WIP 
Jeżeli dobrze pamiętam, to plan tutaj jest taki, żeby zrobić najpierw coś w rodzaju MO-GOMEA-Net i/lub MO-P3-Net. Czyli wziąć i wsadzić ten element "Net" do nowoczesnych metod i zbadać jak to lata.

## Things to do:
Preparation:
1. Write NSGA-II algorithm
2. Split into modules from `general approach`:
* encode data module, 
* evaluate daata module,
* GA module,
* exploit data module
4. Write NSGA-II algorithm (using modules)
5. Write modules (like for NSGA-II) for NSGA-net
6. Write NSGA-Net algorithm (using modules). IMPRTANT: We want to have these `Net` module especially.

Main work:
7. Write modules (like for NSGA-II) for GOMEA
8. Write GOMEA algorithm (using modules)
9. Write MO-GOMEA-Net algorithm (using modules). IMPORTANT: these is research task. There is no MO-GOMEA-Net algorithm! MO stands for Multi-Objective. GOMEA stands gor Gene-pool Optimal Mixing Evolutionary Algorithm (https://www.sciencedirect.com/science/article/abs/pii/S2210650217304765). Net stands for... `WIP`

In the future:
10. Write modules (like for NSGA-II) for P3
11. Write P3 algorithm (using modules)
12. Write MO-P3-Net algorithm (using modules). IMPORTANT: these is research task. There is no MO-P3-Net algorithm! MO stands for Multi-Objective. P3 stands for ... `WIP`. Net stands for... `WIP`



# Usefull links:
* OptFrame - framework for GA experiments - http://git.przewozniczek.eu/przewooz/OptFrame
* NSGA-Net - https://github.com/ianwhale/nsga-net
* hyperparamiters of neural networks - good explanation https://towardsdatascience.com/parameters-and-hyperparameters-aa609601a9ac 
* pymoo - python librarie for multi-object optimalization - https://github.com/anyoptimization/pymoo