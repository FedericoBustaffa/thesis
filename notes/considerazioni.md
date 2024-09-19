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
alle strutture dati che, nel caso in cui
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
