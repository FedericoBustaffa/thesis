# Parallelizzazione di algoritmi genetici

Ogni algoritmo genetico si compone dei seguenti passi:

1. Generazione della popolazione
2. Valuatazione
3. Selezione
4. Crossover
5. Mutazione
6. Valutazione dei nuovi individui
7. Eventuale (se non necessario) rimpiazzo dei vecchi
   individui con fitness peggiore.

Non conosco così bene gli algoritmi genetici da capire
quali siano le dinamiche più corrette e quali siano
eventuali errori da evitare che andrebbero a snaturare
l'algoritmo originale.

## Generazione della popolazione

Non so se si possa veramemte parallelizzare questa fase in
quanto la generazione casuale degli individui probabilmente
richiede un generatore unico in tutto il programma.

Se così fosse si perde solo tempo a sincronizzare ogni thread
sul generatore e sarebbe molto più veloce fare tutto in modo
sequenziale.

Sarebbe da approfondire l'argomento generatori casuali
soprattutto in ambito parallelo.

## Valutazione

Ogni individuo deve essere **valutato** dalla funzione di
fitness ma la valutazione di un individuo non dipende
dalla valutazione di nessun altro individuo.

Possiamo quindi valutare ogni individuo in modo asincrono.
In particolare possiamo pensare di saturare ogni core
disponibile con un thread e dividere equamente il carico
tra ognuno dei thread.

Sia $N$ il numero di individui da valutare e sia $C$ il
numero di core della macchina possiamo creare $C$ thread
o processi, a ciascun dei quali viene dato il compito di
valutare $N / C$ individui.

## Selezione

Per quanto riguarda la **selezione** non so quanto sia
corretto effettuare una parallelizzazione e nel caso non
so quale sai il modo correto di farla. Dobbiamo inoltre
considerare che al momento della selezione gli individui sono
stati valutati e ho pensato quindi a due possibili scenari:

1. Per semplicità poniamoci nella casistica in cui abbiamo
   2 thread. La popolazione viene divisa in due: i "migliori"
   e i "peggiori". Ad ogni thread viene assegnato un gruppo
   che si occupa di effettuare la selezione sul suo gruppo.

   Potrebbe non essere ottimale per garantire la diversità
   all'interno della popolazione.

2. I gruppi vengono creati in maniera random inserendo ogni
   individuo casualmente in uno dei gruppi disponibili. In
   alternativa si potrebbe manipolare la probabilità di
   finire in un gruppo piuttosto che in un altro in base
   al valore di fitness di ogni individuo.

## Crossover

Una volta selezionati i migliori individui dobbiamo formare
gli accoppiamenti tramite l'operatore di **crossover**.
Anche qui non so quanto e se sia giusto parallelizzare il
processo.

Similmente al punto precendente possiamo dividere la
popolazione che ha superato la selezione in gruppi creati

- Del tutto casualmente
- Tramite probabilità basata sulla fitness di modo che
  individui migliori abbiano maggiori probabilità di
  accoppiamento.

Anche qui non so quale dei due casi sia migliore, se
entrambi siano sbagliati o se concettualmente sia proprio
un errore cercare di trovare un versione parallela di
questo step dell'algoritmo.

## Mutazione

Per quanto riguarda la **mutazione** dobbiamo operare su
tutta la popolazione generata nella fase di crossover ma
possiamo tranquillamente applicare la funzione di mutazione
ad ogni individuo delle popolazione in maniera del tutto
indipendente dagli altri individui.

### Parallelizzazione semplice

In questo caso la cosa più banale che mi viene in mente è
di dividere equamente la popolazione tra i thread ed
applicare l'operatore di mutazione ai vari gruppi.

### Pipeline

Una volta generato un individuo nella fase di crossover
non vedo ragione per non mandarlo subito alla fase di
mutazione. In questo modo evitiamo di attendere che l'intera
fase di crossover sia finita.

Ho pensato quindi di costruire una sorta di meccanismo di
più pipeline in parallelo che attendono uno o più individui
alla volta da poter mutare.

## Valutazione dei nuovi individui

Qui valgono praticamente le stesse considerazioni fatte
per la mutazione. Ogni individuo può essere valutato in
modo del tutto indipendente dagli altri, di conseguenza
possiamo

- Assegnare gruppi di individui ai thread che poi saranno
  incaricati di effettuare la valutazione parallelamente
- Estendere la pipeline precedentemente menzionata.

Il risultato finale sarebbe una pipeline a tre step:
Crossover, Mutazione e Valutazione di uno o più individui.

## Rimpiazzo

Per quanto riguarda il rimpiazzo non so se sia possibile
fare qualcosa in parallelo. Mi pare uno step che richiede
di considerare l'intera popolazione per capire quali
individui siano da rimpiazzare o meno.

Sempre che non si accetti un rimpiazzo probabilmente meno
corretto e che potrebbe snaturare il comportamento originale
dell'algoritmo che prevede anche qui la divisione in gruppi
della popolazione (vecchia e nuova). Il risultato sarebbe
che l'algoritmo opera correttamente sui gruppi ma non come
sperato sull'intera popolazione.

## Considerazioni finali

Ognuno di questi passi si basa sulla poca conoscenza e sulla
poca esperienza che ho in campo algoritmi genetici.

### API e comportamento

Come è possibile vedere, in diversi punti, per avere la
possibilità di parallelizzare il più possibile si decide
di non operare più sull'intera popolazione ma su alcuni
gruppi. Se ad esempio la selezione fosse pensata per operare
sull'intera popolazione, il risultato sarebbe probabilmente
diverso da quello sperato.

Ecco che vorrei capire se dare il controllo al programmatore
(e in caso quanto controllo) il quale deve decidere se
parallelizzare ogni fase e nel caso (a seconda della fase)
fornire qualche indicazione al modulo per sapere ad esempio
come formare i vari gruppi.

L'API si complica leggermente ma forse ci si aprono più
scenari e possibilità. Se inoltre l'idea originale prevede
in qualche modo la divisione della popolazione in gruppi,
si avrebbe un supporto _nativo_ da parte del modulo.

### Multi nodo?

Si deve pensare ad un'architettura multi-nodo? Quindi con
anche un sistema di socket o simili?

In generale ci sarebbe anche da capire l'entità dei sistemi
su cui poi tutto ciò girerà dato che per ottenere un
effettivo fattore di accelerazione è necessario ridurre al
minimo lo scheduling dei thread e tenerli sempre a regime.

Il numero di core fisici è dunque un fattore chiave per
far sì che tutto questo funzioni efficientemente.
