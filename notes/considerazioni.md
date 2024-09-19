# Considerazioni

Di seguito alcune considerazioni e dubbi venuti fuori durante
la fase di implementazione del modulo.

## Crossover

- Per la fase di crossover abbiamo bisogno di generare
  le coppie in anticipo nel caso si vogliano estrarre gli
  individui dall'intero gruppo dei selezionati.
- Se il numero di genitori è diverso da 2 e il modo in cui
  le coppie vengono formate può essere definito dall'utente
  allora è necessario:
  - Aggiungere una funzione in cui si definisce come le
    coppie vengano formate. Comporta una separata definizione
    di come le coppie vengano formate e di come agisca
    l'operatore di crossover. Se si definisce tutto insieme
    non si riesce ad ottenere un buon parallelismo a meno di
    suddividere la popolazione in subsample.

## Memoria

Principali considerazioni da fare sono riguardo alle strutture
dati che, nel caso in cui ridimensionabili potrebbero comportare
un approccio differente nell'implementazione.

Questo sia in termini di strutture che contengono gli individui
sia in termini di (forse) lunghezza dei cromosomi. Non mi è
chiaro se un cromosoma possa avere lunghezza variabile o se
tutti i cromosomi debbano avere lunghezza uguale e fissata.

Da considerare anche il fatto che un cromosoma potrebbe essere
formato da elementi di tipo qualsiasi come classi definite
dall'utente. Queste classi potrebbero avere dentro oggetti che
incrementano dinamicamente la loro dimensione?

Parte delle ottimizzazioni che non riguardano il lavoro in
parallelo si basano sul riuso di memoria già allocata. Ecco
diventa necessario capire come si comportano tali strutture
dati.
