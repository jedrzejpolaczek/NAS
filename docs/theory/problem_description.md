# Temat: 
"Ewolucyjna optymalizacja zbiorów treningowych w połączeniu z hiperparametrami sieci głębokich przeznaczonych do analizy obrazu"

# Cechy problemu:

## Zbiór treningowy:
- **wielkość zbioru treningowego**
- **balans między klasami w zbiorze treningowym** (które obrazy mają trafić do zbioru testowego, a które nie)

## Hiperparametry modelu:

| Parametr               | Typ         | Zakres  | Uwagi                                                                                  |
|------------------------|-------------|---------|----------------------------------------------------------------------------------------|
| Liczba warstw          | Całkowity   | 1-n     | Głębokość sieci, od 1 do nieskończoności                                                |
| Liczba filtrów         | Całkowity   | 1-n     | Kernel w warstwach konwoluncyjnych (określa liczbę różnych cech które sieć może wykryć) |
| Rozmiar filtrów        | Całkowity   | 1-n     | Dwie wartości tworzące macierz kwadratową                                               |
| Stride                 | Całkowity   | 1-n     | Krok - jak szybko filtr przesuwa się po obrazie                                         |
| Padding                | Całkowity   | 0-1     | Dodawanie pikseli jak filtr musiałby wyjść poza obraz. 0-1 jako wartości True/False      |
| Funkcja aktywacji      | ?           | ?       | Istnieje ograniczona liczba takich funkcji, więc zakres to tyle ile dopuścimy do wyboru |
| Funkcja straty         | ?           | ?       | Jak liczymy błąd, np. średni błąd kwadratowy (MSE)                                      |
| Optymalizator          | ?           | ?       | Jak aktualizujemy wagi, np. gradient → SGD                                             |
| Pooling                | Całkowity   | 0-n     | Warstwy w których redukujemy wymiary przestrzenne danych                                |
| Batch size             | Całkowity   | 1-n     | Liczba próbek danych przetwarzanych w jednym kroku. Raczej wielokrotności 2              |
| Dropout                | Całkowity   | 0-n     | Technika "wyłączająca" losowo wybrane neurony aby uniknąć przeuczenia                    |
| Early stopping         | Całkowity   | 0-1     | Wczesne zatrzymanie modelu aby go nie przeuczyć. 0-1 jako wartości True/False            |
| Inicjalizacja wag      | ?           | ?       | Można losowo, można ustawić na jedną wartość itd.                                       |
| Liczba epok            | Całkowity   | 1-n     | Ile razy ma zostać powtórzony krok trenujący                                            |
| Szybkość uczenia       | Rzeczywisty | 0-n     | Możliwe, że będzie to średnia z iluś pomiarów                                           |

## Liczba parametrów:
- Rozmiar i złożoność sieci neuronowej - np. rozmiar filtrów w warstwach konwolucyjnych
- Parametry algorytmu genetycznego - zakładam, że osobnik będzie reprezentował hiperparametry sieci neuronowej (**DO PRZEMYŚLENIA: czy osobnik nie powinien zawierać też parametry zbioru**)
- Złożoność danych treningowych **(DO PRZEMYŚLENIA: jak wielkie mają być to obrazy i jak wiele cech mają zawierać, kwestia praktyczna**)

## Budżet obliczeniowy:
TBD - do ustalenia

## Cechy danych:
- Wielkość zbioru i balans między klasami, dochodzi tutaj wiele innych cech jak ich zróżnicowanie, zaszumienie itd. Jest to powód, dla którego zastanawiam się, czy nie odrzucić z tematu tej części o zbiorach.

## Odporność na zakłócenia:
TBD - Do ustalenia (+ przypominajka, że chodzi o to jak bardzo drobna zmiana w parametrach konfiguracyjnych metody wpływa na jej wynik.)

## Słabe punkty optymalizacji:
- Zależność od danych - wyniki sieci neuronowych są bardzo zależne od danych
- Optymalizacja wielokryterialna
- Dobór hiperparametrów

## Zwrócić uwagę na:
- Dobór odpowiedniej metody walidacji uzyskanych wyników
- Złożoność obliczeniowa

# TBD
## Do zrobienia:
- Zobaczyć, jak rozwiązuje się podobne problemy obecnie
- Zwrócić uwagę na to, jakiego rodzaju zbiory są wykorzystywane (cechy obrazów)
- Ustalić budżet obliczeniowy (np. 1000 razy liczony fitness czy coś podobnego, tak by było wymierne dla innych algorytmów których się używa)
- Po wybraniu swojej metody ustalić jej odporność na zakłócenia

### NOTATKA:
- To co robię to evolutionary machine learning

### TODO:
- Lista state-of-art metod
