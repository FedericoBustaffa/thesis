# Parallelizzazione di algoritmi genetici

Ogni algoritmo genetico si compone dei seguenti passi:

1. Generazione della popolazione.
2. Valuatazione della fitness degli individui.
3. Selezione dei migliori individui.
4. Crossover.
5. Mutazione.
6. Si riparte dal punto 2.

Come possiamo vedere l'algoritmo è sequenziale, non
possiamo cioè utilizzare un meccanismo di **pipeline**
per riuscire a far avvenire i vari step in parallelo
perché ognuno dipende dal precedente ma soprattutto
ogni ciclo dipende dal precedente.

Tuttavia è possibile pensare a come parallelizzare
alcuni degli step internamente.

## Parallelizzazione della valutazione

Ogni individuo deve essere valutato dalla funzione di
fitness ma la valutazione di un individuo non dipende
dalla valutazione di nessun altro individuo.

Possiamo quindi valutare ogni individuo in modo asincrono.
