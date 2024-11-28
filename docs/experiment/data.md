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