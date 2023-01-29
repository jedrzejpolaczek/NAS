# Why this repo exist?
# What is NAS and E-NAS?
# E-NAS
Experiments about an evolutionary approach to Neural Architecture Search (E-NAS)
# How to use this repo
## File structure
* models - implementation of models and thier templates to use for search startegies.
* search space - (encoding) way of represent search space for example how we can code neural networks layers to be used and modified by search strategies. Most of the time implementation of it will be individuals and population for genetic algorithm.
* search strategies - ways to find solution from search space for example genetic algorithms or reinforcement learning.
* examples - mostly contains jupyter notebooks with examples and explanations how each algorithm works.


# Usefull links and influences:
* https://github.com/gmontamat/gentun (models implementation and general idea how genetic algorithms should be implemented)
* https://github.com/anonymone/Neural-Architecture-Search (well structured knowledge about NAS)
* https://github.com/adam-katona/NSGA_2_tutorial (idea about jupyter notebooks for algorithms and help to understand NSGA2)
* https://www.researchgate.net/figure/Flow-chart-of-step-by-step-procedure-of-NSGA-II-algorithm_fig3_261922808
* https://github.com/ianwhale/nsga-net (starting point of this work)
# Notes:

Założenia:
    1. ML to przeksztalcenia geometryczne z przestrzeni A do B
    2. ML korzysta z przestrzeni hipotez/możliwości (zdefiniowany zbiór operacji)
        2.1 Topologia sieci definiuje przestrzen hipotez
        2.2 Od parametrów topologi sieci (np. ilosc neuronow) zalezy wydajnosc/sprawnosc(?) sieci
    3. Hiperparametry można zoptymalizować (jest to przestrzeń ciągła(?))
    4. Pod co optymalizujemy? Błąd walidacji czy coś jeszcze? Dokładność?
        4.1 Czy można zrobić NAS bez trenowania sieci?

Wady i zalety algorytmów:
    EA:
        Ogólnie:
            Założenie:
                
            Wady:
                1. Potrzebuje wykonać dużo obliczeń -> jest wolny
            Zalety:
                1. Dobrze się sprawdza w przeszukiwaniu przestrzeni rozwiązań
        NSGA
            Założenie:

            Wady:
                1. 
            Zalety:
                1. 
        NSGA-II
            Założenie:

            Wady:
                1. 
            Zalety:
                1.             
    RL:
        Założenie:
            
        Wady:
            1. łatwo wpada w lokalne optimum
        Zalety:
            1. szybko znajduje lokalne optimum "reagując" na to co się dzieje

Problemy:
    Ogólnie:
        1. Każdy algorytm (EA, RL, itd.) potrzebuje parametrów wejściowych więc mogliobyśmy używać w kółko algorytmów żeby dopasować idealne parametry dla nich itd.
    EA:
        1.
    RL:
        1.

Wykorzystanie w biznesie:
    1. RAPIDS on AWS Sagemaker
    2. Vertex AI NAS - https://cloud.google.com/blog/products/ai-machine-learning/vertex-ai-nas-makes-the-most--advanced-ml-modeling-possible
