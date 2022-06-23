# E-NAS
<<<<<<< HEAD
Experiments about an evolutionary approach to Neural Architecture Search (E-NAS)
# E-NAS
=======
>>>>>>> e11fcc3548d7c7fa55d9b226930ac4c07ad20cb0
Experiments about an diffrent approach to Neural Architecture Search (NAS).

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
<<<<<<< HEAD
        Ogólnie:
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
        NSGA-Net:
            Założenie:
                
            Wady:
                1. 
            Zalety:
                1. 
    RL:
        Założenie:
            
=======
        Wady:
            1. 
        Zalety:
            1. 
    RL:
>>>>>>> e11fcc3548d7c7fa55d9b226930ac4c07ad20cb0
        Wady:
            1. 
        Zalety:
            1. 

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
