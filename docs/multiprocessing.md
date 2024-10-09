# Analisi multiprocessing

Di seguito alcune considerazioni sull'utilizzo
del multiprocessing per la parallelizzazione di
un algoritmo genetico.

## Motivazioni

L'utilizzo del modulo `multiprocessing` è ciò che
viene considerato lo standard per il calcolo
parallelo in Python, soprattutto se si parla di
task CPU bound.

Permette di aggirare il problema introdotto dal
GIL, il quale non permette il classico paradigma
multithread, possibile in altri linguaggi come C o
C++.

Tramite multiprocessing è dunque possibile non
scendere a compromessi in quanto ad espressività
del codice. Il programmatore è libero di scrivere
il suo algoritmo genetico utilizzato tipi e
strutture dati native di Python e, se necessario
può ricorrere a librerie di terze parti senza
problemi.

### Problematiche

Il multiprocessing di contro non è la cosa più
leggera che ci sia. Di certo non è possibile
pensare ad un approccio in cui i processi vengono
creati e distrutti ad ogni iterazione, in quanto
si genererebbe del dell'overhead sicuramente non
trascurabile.

La scelta di creare un pool di processi worker che
rimane in vita dall'inizio alla fine
dell'esecuzione è dunque una scelta quasi obbligata
e che introduce la necessità di un qualche
meccanismo di sincronizzazione per avviare e
mettere in attesa i processi.

L'altro potenziale limite è l'assenza di memoria
condivisa. Non è dunque possibile avere strutture
dati condivise tramite puntatori o definite
globalmente a cui ogni processi può accedere. Anche
qui la scelta di condividere parti della struttura
dati tramite qualche meccanismo di streaming dati
è praticamente obbligata.

#### Memoria condivisa

Esiste la possibilità di creare un blocco di
memoria condivisa che risiede al di fuori di ogni
worker e al quale è possibile accedere in modo
diretto come si farebbe tramite multithreading.

Ho trovato tuttavia questo meccanismo molto
limitante per l'espressività che vorrebbe in
qualche modo garantire la libreria. Il motivo è
che per riuscire a accedere direttamente al blocco
di memoria lo si deve fare tramite oggetti che
supportano il **Buffer Protocol** di Python, come
ad esempio _numpy array_ o _bytearray_.

Questo limita molto i tipi e le strutture dati
che si vogliono impiegare e ci spinge ad un
approccio più classico in cui possiamo manipolare
solo array numerici o simili.

Non sarebbe quindi possibile avere cromosomi
dalla struttura complessa ma soprattutto
personalizzata.

## Modello di calcolo parallelo

Il modello di calcolo proposto si offre di operare
in parallelo nelle fasi di crossover, mutazione e
valutazione.

Una volta selezionati gli individui per la
riproduzione, si suddivide la lista che li contiene
in $W$ chunk uguali, dove $W$ è il numero di
processi worker. Ciascun chunk viene poi inviato
ai worker.

Per la condivisione ho optato per un meccanismo
basato su code (`multiprocessing.Queue`) di
comunicazione. Ogni processo possiede due code, una
per la ricezione dati, l'altra per l'invio.

Le code permettono di implementare in modo molto
semplice il paradigma _produttore-consumatore_,
fornendo due metodi principali (`put` e `get`)
che permettono rispettivamente di

- Inserire un elemento nella coda. Se questa è
  piena il processo si blocca finché non vi è
  uno slot libero.
- Estrarre un elemento dalla coda. Nel caso in cui
  questa sia vuota ci si blocca in attesa che un
  elemento venga inserito.

Per velocizzare ulteriormente l'invio e la
ricezione dati ho fatto uso della libreria
`asyncio`, la quale, tramite la sintassi
`async`/`await` permette di effettuare operazioni
I/O bound e modo asincrono.

## Prestazioni

Il modello di calcolo, per quanto semplice, sotto
le giuste ipotesi offre lo speed up sperato (o
comunque ci si avvicina molto) quando almeno una
delle tre fasi parallelizzate rappresenta la parte
più pesante dell'algoritmo.

I test svolti prendono come esempio una popolazione
di $10.000$ individui, tutti identificati da
cromosomi composti da $200$ interi.

Ogni individuo occupa uno spazio di $48$ byte e per
inserirlo nella coda sono necessari in media
$0.9 \; \mu s$. Per inserirne $10.000$ sono stati
necessari circa $15 \; ms$.

Per elaborare un individuo mediamente sono necessari
$0.6 \; ms$ secondi ed è quindi

<!-- Ricontrollare gli ordini di grandezza -->
Se la valutazione della fitness di ogni individuo è
abbastanza pesante da coprire l'overhead dovuto
alla condivisione di quell'individuo con il worker
di anche solo un ordine di grandezza temporale, lo
speed up sarà quello sperato (in particolare vicino
ad un fattore $W$).
