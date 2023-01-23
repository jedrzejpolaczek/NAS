# E-NAS
Experiments about an evolutionary approach to Neural Architecture Search (E-NAS)
# E-NAS
Experiments about an diffrent approach to Neural Architecture Search (NAS).

# File structure
* models - implementation of models and thier templates to use for search startegies.
* search space - way of represent search space for example how we can code neural networks layers to be used and modified by search strategies.
* search strategies - ways to find solution from search space for example genetic algorithms or reinforcement learning.
* utils

TODO: 
* Add generall information about NAS, diffrent approaches and so on (in context of hyperscalar optimization and automation ML)
* Add "classic" NAS algorithms implementation + experiments (jupyter notebooks) diveded by class of algorithms
* Add proper cons and pros to each 
* Add assumptions
* Add ideas?

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
