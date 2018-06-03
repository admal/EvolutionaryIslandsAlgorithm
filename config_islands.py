# prawdopodobienstwo mutacji - float - prawdopodobienstwo czy dany
# osobnik w populacji bedzie mutowany
MUTATE_PROB = 0.2
# wspolczynnik mutacji - float - prawdopodobienstwo czy dany gen w
# wybranym do mutacji osobniku zostanie zmutowany
MUTATE_COEF = 0.1
# prawdopodobienstwo rekombinacji - float - prawdopodobienstwo czy
# dany osobnik bedzie rekombinowany z innym
RECOMB_PROB = 0.2
# rozmiar populacji - natural number - ilosc osobnikow w populacji
POPULATION_SIZES = [300, 100, 200]
# liczba epok - natural number - liczba generacji nowych populacji potomnych
MAX_ITERS = 10

TOURNAMENT_SIZE = 50
# wielkosc migracji - float - jaka czesc populacji zostanie przeniesiona do
# innej populacji
MIGRATION_SIZE = 20
# interwal migracji - natural number - liczba epok, ktora musi minac od
# ostatniej migracji, zeby nastapila nowa migracja
MIGRATION_INTERVAL = 20
# topologia polaczen - graph - graf polaczen gdzie wierzcholki to podpopulacje,
# a krawedzie ozanczaja czy z jednej podpopulacji osobniki moga
# zmigrowac do innej podpopulacji
TOPOLOGY = [[2],[0, 2],[1]]
# length of a genome
GENOME_LEN = 2



