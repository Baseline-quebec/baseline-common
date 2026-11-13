# Règles de revue

Ces règles servent de grille à toute revue de code de baseline-common et des dépôts spécialisés
(baseline-optimization, baseline-ml, baseline-llm), qui n'en gardent pas de copie. Une revue cite
leur numéro. Dans un dépôt spécialisé, les dossiers du module, à la racine du paquet, tiennent le
rôle de `core/`.
Chaque règle tient dans son titre en gras ; le texte qui suit l'explique et donne un exemple.
Une règle marquée *(signal)* n'interdit rien : elle pose une question à la revue, et un écart
justifié est accepté.

Un module qui a des règles propres les porte dans sa documentation : celles du moteur
d'optimisation sont dans `docs/optimization-engine.md` de baseline-optimization (« Règles du moteur »).

## Conception

1. **Le constructeur reçoit ce qui configure l'objet, les méthodes reçoivent les données.**
   - Le constructeur reçoit ce qui est donné une fois : réglages, dépendances, paramètres.
   - Une méthode reçoit les données qu'elle traite. Un argument qui ne sert qu'à une méthode reste
     un argument de cette méthode.
   - Ce qui ne vaut que pendant un appel (un drapeau, un curseur, la forme en cours de lecture) est
     un argument ou une variable locale, jamais un attribut.
   - Exemple : `PostgresRepository(session, Machine, scope=scope)`, puis `insert_many(records)`.
2. **On instancie une classe pour la configurer.** Les arguments du constructeur (réglages,
   dépendances, paramètres) sont ce qui configure l'objet. Une classe dont le constructeur ne
   prendrait aucun argument n'a rien à configurer, donc ne s'instancie pas :
   - une logique sans état est une fonction du module ;
   - une façon de construire une instance est une méthode de classe de la classe construite
     (`Problem.from_files(path)`) ;
   - un travail qui a besoin d'un état pendant un appel devient une méthode de la classe qui
     l'appelle.
3. **Une classe a une seule responsabilité.** Une classe qui en mêle deux (lire des plans et écrire
   des règles, par exemple) est coupée en deux. Une méthode qui n'a rien à faire dans sa classe en
   sort.
4. **Un contrat s'écrit en `ABC`.**
   - Les implémentations héritent de l'`ABC` : le contrat est explicite, et un oubli est signalé dès
     l'instanciation.
   - Une `ABC` a **plusieurs vraies sous-classes** et une raison de les réunir (un comportement
     commun, un appelant qui les prend toutes). Un faux d'essai ne compte pas comme sous-classe.
   - Une seule implémentation : pas d'abstraction.
   - Un `Protocol` demande une raison forte : décrire un objet qu'on ne peut pas faire hériter, comme
     une classe d'une bibliothèque tierce. Nos propres classes héritent toujours d'une `ABC`.
   - Les patrons connus s'appliquent sous leur nom (Repository, Gateway, Strategy, Adapter) plutôt
     qu'une construction maison : le nom dit au lecteur ce que fait la classe avant qu'il lise son
     code.
5. **Une valeur se vérifie une fois, quand l'objet est construit.**
   - Par un `model_validator` Pydantic, ou par le constructeur d'une classe simple.
   - Figer un enregistrement (`frozen=True`) est encouragé : une valeur qui ne change plus après sa
     construction (une configuration, un résultat, une entrée lue) reste valide. On ne fige pas un
     enregistrement dont le rôle est d'être modifié en place (un état mis à jour pas à pas) ;
     `validate_assignment=True` revérifie alors chaque modification.
   - Une propriété rend une valeur ; elle ne revérifie pas un type à chaque lecture.

## Nécessité

6. **Tout ce qui existe sert.** Chaque module, classe, méthode, champ, paramètre et exception est
   nécessaire ; ce qui n'est ni utilisé ni testé est retiré. Une enveloppe sans comportement ni
   information propre est de la sur-ingénierie. Par exemple, une exception maison vide
   (`UnsupportedError`) cède la place à une exception standard (`ValueError`).
7. **Une seule façon de faire chaque chose.**
   - Pas de `@overload`, pas de signature « une chose ou une liste », pas de deux méthodes publiques
     pour le même effet.
   - Pas de deuxième implémentation à côté de celle qui existe (une deuxième recherche, un deuxième
     chargement) : on appelle l'existante.
8. **Ni variable globale ni enveloppe triviale.**
   - Les constantes vivent dans la classe qui s'en sert.
   - Pas de lambda ni de fonction d'une ligne qui ne fait que relayer un appel ou renommer une
     expression.
   - Seule exception : une lambda qui choisit un champ du schéma dans une déclaration
     (`model.param(lambda op: op.duration, operations)`), pour que l'IDE vérifie le chemin.

## Organisation

9. **Le code de `core/` est générique.** Il ne nomme ni un client, ni un problème, ni une entité
   métier. Un besoin propre à un problème passe par un paramètre ou une abstraction générique,
   jamais par du code qui nomme ce problème. Ce qui est propre à un problème ne sert qu'aux essais.
10. **Aucun nom de classe ne commence par un souligné.** Une classe intermédiaire, comme l'état d'une
    exécution, vit dans le fichier de la classe qui s'en sert, sous un nom normal (`PrecheckRun`, pas
    `_Run`).
11. **Un script est un point d'entrée court qui montre ce que les essais ne montrent pas** : une
    mesure, un banc, un usage complet. Il vit dans `scripts/`, avec ce qui ne sert qu'à lui ; rien de
    tout cela ne vit dans `tests/`. Un script qui ne montre rien de neuf est retiré.

## Écriture

12. **Une méthode sans argument devient une propriété seulement si sa lecture est sans surprise.**
    - Une propriété se lit comme un attribut : le lecteur suppose qu'elle est immédiate, sans effet et
      qu'on peut la lire dix fois sans coût.
    - Elle reste une méthode dès qu'un doute existe : un calcul coûteux, une entrée-sortie (fichier,
      réseau, base), un effet de bord, un résultat qui change d'un appel à l'autre, ou un `async`. Les
      parenthèses de l'appel montrent alors qu'il y a un travail.
    - Exemple : `task.end` (début + durée) est une propriété ; `schedule.render()` (coûteux) et
      `clock.now()` (change à chaque appel) restent des méthodes.
13. **Plus de quatre arguments : se demander s'il y en a trop.** *(signal)* `self` non compté,
    constructeur compris. Souvent, des arguments qui vont ensemble forment un enregistrement, ou la
    méthode fait trop de choses et se coupe. Si chacun est une donnée distincte et nécessaire, ils
    restent.
14. **Des listes, pas des tuples.** Un tuple demande une raison : une valeur qui doit être immuable
    ou hachable (une clé de dictionnaire, un membre d'ensemble). Sinon une liste, un dictionnaire ou
    un enregistrement.
15. **Les commentaires sont rares et courts** : une ou deux lignes qui disent une contrainte que le
    code ne montre pas. Aucun séparateur de section.
16. **Une docstring tient en une ligne,** plus une ligne par entrée `Args`, `Returns` et `Raises`. Dans
    une docstring ou un document, une méthode s'écrit avec ses arguments (`run(bound)`) ou sans
    parenthèses (`run`), jamais `run()`.
17. **Le code respire.**
    - Une ligne vide avant et après chaque bloc (`if`, `for`, `while`, `with`, `try`), sauf au début
      ou à la fin du bloc qui le contient.
    - Une ligne vide entre les étapes d'une méthode.
    - Un essai sépare préparation, action et vérification par une ligne vide.

## Essais

18. **Ce que plusieurs essais préparent de la même façon est une fixture.**
    - Un essai reçoit une fixture par son nom, en argument ; il ne l'importe jamais.
    - Une classe d'aide aux essais (un faux serveur, un modèle d'essai) vit dans un module du dossier
      d'essais et s'importe de là, jamais d'un `conftest.py`.
    - Une fixture rend une valeur neuve à chaque essai. Une portée plus large (`module`, `session`)
      demande un coût mesuré, et une valeur qu'aucun essai ne modifie.
19. **Les `conftest.py` forment une hiérarchie.** Une fixture vit au plus près des essais qui s'en
    servent : dans le fichier d'essai si un seul s'en sert, sinon dans le `conftest.py` du plus petit
    dossier qui les contient tous. `tests/conftest.py` ne garde que ce qui sert partout. Une fixture
    ne remonte pas « au cas où ».
20. **Un essai respecte les couches du code qu'il teste.** Ses imports suivent le même sens que le
    code : `api → orchestrator → service → data_schema → core`.
    - Un essai sous `tests/core/` n'importe ni `service/`, ni `orchestrator/`, ni `data_schema/`, ni
      `api/`, ni les essais de ces couches.
    - Un essai d'un module de `core/` n'importe pas un autre module de `core/`.
    - Entre dépôts, le sens est le même : un dépôt spécialisé peut lire `core/` ou `data_schema/` de
      baseline-common (`graph/` de baseline-ml lit un `MultiDiGraph`) ; baseline-common ne lit jamais
      un dépôt spécialisé, et deux dépôts spécialisés ne se lisent pas.

## Erreurs et journaux

21. **Une exception remonte jusqu'au code qui sait la traiter.**
    - Le code lève une exception précise. Il ne l'attrape que pour s'en remettre ou pour la traduire.
    - Pas de `except` qui avale l'erreur, pas de `try` qui journalise puis relance.
    - La traduction en réponse HTTP se fait une seule fois, dans `ExceptionHandlerMiddleware`. Un
      service ou un repository ne connaît pas HTTP.
    - Un `except Exception` n'existe qu'à une frontière finale (une requête, un processus enfant), sa
      raison écrite dans son `noqa`.
22. **On journalise avec `loguru`, jamais avec `print`.** Une erreur est journalisée une seule fois, là
    où elle est traitée (`logger.exception`). Aucun secret ni donnée personnelle dans un journal.
