# Considerazioni

Di seguito alcune considerazioni e dubbi venuti
fuori durante la fase di implementazione del
modulo.

## Crossover

- Per la fase di crossover abbiamo bisogno di
  generare le coppie in anticipo nel caso si
  vogliano estrarre gli individui dall'intero
  gruppo dei selezionati.
- Se il numero di genitori è diverso da 2 e il modo
  in cui le coppie vengono formate può essere
  definito dall'utente allora è necessario:
  - Aggiungere una funzione in cui si definisce
    come le coppie vengano formate. Comporta una
    separata definizione di come le coppie vengano
    formate e di come agisca l'operatore di
    crossover. Se si definisce tutto insieme non
    si riesce ad ottenere un buon parallelismo a
    meno di suddividere la popolazione in
    subsample.

## Memoria

Principali considerazioni da fare sono riguardo
alle strutture dati che, nel caso in cui siano
ridimensionabili potrebbero comportare un
approccio differente nell'implementazione.

Questo sia in termini di strutture che contengono
gli individui sia in termini di (forse) lunghezza
dei cromosomi. Non mi è chiaro se un cromosoma
possa avere lunghezza variabile o se tutti i
cromosomi debbano avere lunghezza uguale e fissata.

Da considerare anche il fatto che un cromosoma
potrebbe essere formato da elementi di tipo
qualsiasi come classi definite dall'utente. Queste
classi potrebbero avere dentro oggetti che
incrementano dinamicamente la loro dimensione?

Parte delle ottimizzazioni che non riguardano il
lavoro in parallelo si basano sull'utilizzo
efficiente di memoria già allocata. Ecco diventa
necessario capire come si comportano tali strutture
dati.

- Per esempio quando vengono generati nuovi
  individui non viene ricreata da zero la struttura
  dati che li contiene per poi rimpiazzare la
  vecchia. Ho riscontrato un significativo
  miglioramento delle performance se vado
  semplicemente a rimpiazzare i vecchi individui
  uno per uno con i nuovi.
- Di norma non sarebbe un problema avere una
  struttura dati che cambia di dimensione ma in
  questo caso la struttura dati è condivisa e la
  dimensione della memoria allocata per contenerla
  è fissata e non può (a quanto ho capito) cambiare
  dinamicamente. Nel caso in cui tale struttura
  dovesse crescere di dimensioni si dovrebbe
  riallocare anche la memoria condivisa e copiare
  i dati.

### Cromosommi complessi

Per quanto riguarda l'allocazione in memoria
condivisa di cromosomi definiti dall'utente come
oggetti complessi ci sono diversi problemi.

Da quanto sono riuscito a capire, tutti i
meccanismi di condivisione dati in uno spazio di
memoria condiviso, sono molto limitati ai tipi
base di Python e a strutture dati molto semplici.

Per esempio è possibile condividere una lista di
valoro numerici ma non una lista di liste (per
esempio).

Ecco che l'idea di avere un cromosoma definito come
segue diventa abbastanza complicata.

```py
class Chromosome:
  def __init__(self, chromosome, fitness):
    self.chromosome = chromosome
    self.fitness = fitness
```

Questo perché il parametro `chromosome` dovrebbe
essere una lista o un array di un tipo
potenzialmente qualsiasi. Il problema è che poi
sarebbe impossibile creare una lista di condivisa
di oggetti `Chromosome` a meno che non si aggiunga
un po' di lavoro in più da dare al programmatore.

In particolare il programmatore dovrebbe modellare
la sua classe che rappresenta il gene in modo che
diventi un `ctype`.

Dobbiamo anche tenere di conto che per oggetti
troppo complessi sarebbe poi necessario usare
strumenti tipo `pickle` per serializzare e
deserializzare i dati. Questo però elimina il
vantaggio della memoria condivisa in quanto si
creerebbero delle coppie ogni volta.

## Memoria condivisa vs Pipe

Non sembrano esserci differenze sostanziali nei
due approcci anche se in teoria la memoria
condivisa dovrebbe essere molto più veloce.

I meccanismi di sincronizzazione offerti dal
modulo multiprocessing sembrano avere un overhead
talmente alto da far sì che si perda tutto il
vantaggio guadagnato dalla memoria condivisa.

Vorrei capire se sto usando la memoria condivisa
nel modo corretto o se inavvertitamente genero
nuovi oggetti.
