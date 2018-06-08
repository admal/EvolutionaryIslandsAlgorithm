# prawdopodobienstwo mutacji - float - prawdopodobienstwo czy dany
# osobnik w populacji bedzie mutowany
MUTATATION_PROB = 0.2
# wspolczynnik mutacji - float - prawdopodobienstwo czy dany gen w
# wybranym do mutacji osobniku zostanie zmutowany
MUTATATION_COEF = 0.01
# prawdopodobienstwo rekombinacji - float - prawdopodobienstwo czy
# dany osobnik bedzie rekombinowany z innym
RECOMB_PROB = 0.2
# rozmiar populacji - natural number - ilosc osobnikow w populacji
POPULATION_SIZES = [20, 20, 20, 20]
# liczba epok - natural number - liczba generacji nowych populacji potomnych
MAX_ITERS = 10

TOURNAMENT_SIZE = 6
# wielkosc migracji - float - jaka czesc populacji zostanie przeniesiona do
# innej populacji
MIGRATION_SIZE = 5
# interwal migracji - natural number - liczba epok, ktora musi minac od
# ostatniej migracji, zeby nastapila nowa migracja
MIGRATION_INTERVAL = 5
# topologia polaczen - graph - graf polaczen gdzie wierzcholki to podpopulacje,
# a krawedzie ozanczaja czy z jednej podpopulacji osobniki moga
# zmigrowac do innej podpopulacji
TOPOLOGY = [[1],[2],[3],[0]]
# length of a genome
GENOME_LEN = 2

EPSILON = 1e-08


