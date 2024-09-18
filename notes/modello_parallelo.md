# Modello parallelo

Per il momento la parte di maggiore interesse da
parallelizzare è la fase in cui l'algoritmo compie
il crossover, la mutazione e la valutazione dei
nuovi individui.

Il modello a cui ho pensato è molto semplice ma
dovrebbe ridurre un po' di overhead dovuto ad
eventuali sincronizzazioni tra i processi.

Considerando anche che le tre fasi vengono ripetute
ad ogni iterazione, se si ricreassero i processi
ad ogni iterazione si genererebbe un overhead
troppo grande. Dobbiamo quindi implementare un
soluzione in cui i processi sono attivi per tutta
la durata dell'algoritmo e tramite un qualche
meccanismo di sincronizzazione vengono messi in
atteso o risvegliati.

## Suddivisione del carico

Come suddisivisione del carico di lavoro per il
momento pensavo ad un qualcosa di molto semplice
in cui le strutture dati vengono suddivise
equamente tra i processi in base al numero dei
processi stessi.

Per determinare il numero di processi in gioco
bisogna tenere di conto anche il numero di core
fisici (e logici) che si hanno ha disposizione per
non creare overhead dovuto allo scheduling.

Dato che le tre fasi citate poco fa vengono
eseguite in sequenza, ho pensato che sarebbe
inutile creare un pool di processi per il
crossover, uno per la mutazione e uno per la
valutazione. Per intenderci un qualcosa di
questo genere:

![](images/modello1.svg)

Si aggiungerebbe dell'overhead necessario alla
sincronizzazione dei vari processi che a mio parere
si potrebbe benissimo evitare accorpando le tre
fasi in un'unico processo come mostro di seguito:

![](images/modello2.svg)

In questo modo si hanno solo due fasi di
sincronizzazione, la prima quando si inizia la fase
in parallelo e la seconda quando si termina e si
deve unire i risultati ottenuti per la fase di
rimpiazzo.

Una volta che i processi terminano il loro lavoro
devono essere messi in attesa e risvegliati solo
al momento opportuno.

## Strutture dati condivise

Per capire quali siano le strutture dati condivise
tra i vari processi ho considerato il seguente
modello:

![dataflow](images/dataflow.svg)
