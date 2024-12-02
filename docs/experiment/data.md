Sum up:

|Type|||Description|Example|
|---|---|---|---|---|
|Quantitative data type|||This type of data type consists of numerical values. Anything which is measured by numbers.|profit, quantity sold, height, weight, temperature|
||Discrete data type||The numeric data which have discrete values or whole numbers. This type of variable value if expressed in decimal format will have no proper meaning. Their values can be counted.|No. of cars you have, no. of marbles in containers, students in a class|
||Continuous data type||The numerical measures which can take the value within a certain range. This type of variable value if expressed in decimal format has true meaning. Their values can not be counted but measured. The value can be infinite.|height, weight, time, area, distance, measurement of rainfall|
|Qualitative/Categorical data type|||These are the data types that cannot be expressed in numbers. This describes categories or groups and is hence known as the categorical data type.||
||Structured Data||This type of data is either number or words. This can take numerical values but mathematical operations cannot be performed on it. This type of data is expressed in tabular format.|Sunny=1, cloudy=2, windy=3 or binary form data like 0 or1, Good or bad,|
||Unstructured data||This type of data does not have the proper format and therefore known as unstructured data.|textual data, sounds, images, videos|
|||Nominal Data Type|This is in use to express names or labels which are not order or measurable.|male or female (gender), race, country|
|||Ordinal Data Type|This is also a categorical data type like nominal data but has some natural ordering associated with it.|likert rating scale, Shirt sizes, Ranks, Grades,|
|||Interval Data Type|This is numeric data which has proper order and the exact zero means the true absence of a value attached. Here zero means not a complete absence but has some value. This is the local scale.|Temperature measured in degree Celsius, time, Sat score, credit score, pH, etc. difference between values is familiar. In this case, there is no absolute zero. Absolute|
|||Ratio Data Type|This quantitative data type is the same as the interval data type but has the absolute zero. Here zero means complete absence and the scale starts from zero. This is the global scale.|Temperature in Kelvin, height, weight|


From: https://pianalytix.com/data-types-in-machine-learning/

-----

From: https://trainingdata.pro/data-types-in-machine-learning

Quantitative data types
    Numerical data
        Discrete data
        Continuous data
    Time-series data
Qualitative data types
    Categorical data
    Text data
    Image data
    Audio data
    Video data
Sensor data
Graph data

-----

From: Systematic review of data-centric approaches in artiﬁcial intelligence and machine learning (w references PDF)

Dane strukturalne
Dane niestrukturalne
Dane częściowo strukturalne

-----

Dane z Źródeł:

    Dane tekstowe: Źródła wspominają o przetwarzaniu języka naturalnego (NLP) [1-3]. Przykłady zastosowań to analiza tekstu, tłumaczenie maszynowe, odpowiadanie na pytania [1-3].
    Dane wizualne: Źródła odnoszą się do rozpoznawania obrazów [4, 5]. Sugeruje to wykorzystywanie zdjęć i filmów [4, 5].
    Dane tabelaryczne: Wzmianki o regresji liniowej [6, 7] i uczeniu wzmocnionym [1, 8-14] wskazują na dane w formie tabelarycznej, gdzie kolumny reprezentują różne cechy [1, 6-14].
    Dane relacyjne: Źródła wspominają o sieciach bayesowskich [3, 4, 15, 16] i uczeniu statystycznym modeli relacyjnych [17]. Możliwe jest więc wykorzystywanie danych relacyjnych, np. z baz danych [3, 4, 15-17].

Ogólne Typy Danych w Uczeniu Maszynowym (informacje spoza źródeł):

    Dane numeryczne: liczby całkowite, zmiennoprzecinkowe, wartości ciągłe.
    Dane kategoryczne: dane tekstowe lub numeryczne reprezentujące kategorie, np. płeć, kolor, typ produktu.
    Dane czasowe: dane zebrane w określonych punktach czasowych, np. dane giełdowe, dane pogodowe, sygnały medyczne.
    Dane grafowe: dane reprezentujące relacje między obiektami, np. sieci społecznościowe, sieci neuronowe, mapy drogowe.
    Dane audio: pliki dźwiękowe, np. mowa, muzyka, dźwięki otoczenia.




Poczyńmy założenia:
1. Algorytm musi operować w zdefiniowanym środowisku.
2. Modele ML pracują w zdefiniowanym środowisku
3. Jest wiele zdefiniowanych środowisk dla modeli ML

By zacząć rozwiazywać problem zacnzijmy od zdefiniowania danych na których będziemy pracować. Powinniśmy odpowiedzieć sobie na takie pytania jak
1. Jak identyfikujemy/rozróżniamy zbiory danych


W informatyce wszystko to zbiór 0 i 1 interpretowanych w określony sposób. Każda kolejna interpretacja (tekst, obraz, dźwięk, liczby itd.) to tylko pewnego rodzaju interfejs, który ma za zadanie ułatwić nam komunikacje z komputerem lub uprościć niektóre obliczenia korzystając ze swoistych skrótów myślowych. I tak wykorzystujemy:
1. Typy danych (głównie zaimplementowane w różnych językach programowania)
    1.1  typ całkowity (w C, C++, Javie np. int, w Pascalu np. integer) – typ reprezentujący liczbę całkowitą z jakiegoś zakresu zależnego od języka a nawet konkretnej implementacji.
    1.2 typ zmiennoprzecinkowy (w C, C++, Javie np. double, w Pascalu np. real) – typ reprezentujący przybliżoną wartość liczby rzeczywistej.
    1.3 typ stałopozycyjny (w PL/1, Cobol) – typ reprezentujący liczbę wymierną o stałym mianowniku.
    1.4 typ znakowy (w C, C++, Javie, Pascalu np. char) – typ reprezentujący pojedynczy znak (ASCII lub w nowszych implementacjach Unicode).
    1.5 typ tekstowy (w Javie String, w C++, Pascalu np. string) – typ reprezentujący cały tekst. W C jego rolę pełni tablica znaków, na którą wskazuje wskaźnik do typu znakowego – char*, zakończona znakiem o wartości zero – '\0'.
    1.6 typ wskaźnikowy – oznacza wskaźnik na zmienną zadanego typu.
    1.7 typ referencyjny – odmiana wskaźnika, referencja jest różnie pojmowana w poszczególnych językach.
    1.8 typ wyliczeniowy – typ mogący przyjmować jedną z zadanych symbolicznych wartości, np. (czerwony, zielony, niebieski).
    1.9 typ tablicowy – ciąg zmiennych zadanego typu indeksowanych liczbą naturalną z pewnego przedziału (w większości języków programowania) lub dowolnym unikalnym kluczem który może być zarówno liczbą jak i ciągiem znaków (w PHP).
    1.10 typ strukturalny – zespół połączonych w jedną całość zmiennych zwanych polami struktury. Do każdego pola można się odwoływać oddzielnie.
    klasa (typ obiektowy) – odmiana struktury, w której oprócz zespołu danych dodane są także procedury działających na tych danych. Zmienna typu klasy nazywa się obiektem. Pojęcie klasy spowodowało powstanie nowego paradygmatu programowania, zwanego programowaniem obiektowym i zrewolucjonizowało sposób myślenia programisty, który od tej pory patrzy na program jako na zbiór autonomicznych obiektów.
    1.11 typ pusty (np. void w C i C++) – występuje np. w oznaczaniu funkcji nie zwracających żadnych wartości.
    1.12 typ logiczny (np. bool w C++) – może przyjmować wartości logiczne 1 (true, t) lub 0 (false, nil).
    1.13 typ bitowy reprezentujący ciąg bitów, (np. '01101'B – PL/I).
    1.14 typ zbiorowy reprezentujący zbiory elementów (np. [pon, wt, sr, czw, pt] – Pascal).
    1.15 typ zespolony reprezentujący liczby zespolone.
    1.16 typ etykietowy reprezentujący wartości będące etykietami instrukcji, wskazaniami instrukcji.
2. Struktury danych, np.
    2.1 rekord lub struktura
    2.2 tablica
    2.3 lista
    2.4 stos
    2.5 kolejka
    2.6 drzewo i jego liczne odmiany (np. drzewo binarne)
    2.7 graf
    2.8 kopiec