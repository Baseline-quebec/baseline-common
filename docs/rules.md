# Règles de revue

Ces règles servent de grille à toute revue de code de baseline-core. Une revue cite leur numéro.
Chaque règle tient dans son titre en gras ; le texte qui suit l'explique et donne un exemple.
Une règle marquée *(signal)* n'interdit rien : elle pose une question à la revue, et un écart
justifié est accepté.

Un module qui a des règles propres les porte dans sa documentation : celles du moteur
d'optimisation sont dans `docs/optimization-engine.md` (« Règles du moteur »).

## Conception

1. **Le constructeur reçoit la configuration, les méthodes reçoivent les données.**
   - Le constructeur reçoit ce qui est donné une fois : réglages et dépendances.
   - Une méthode reçoit les données qu'elle traite. Un argument qui ne sert qu'à une méthode reste
     un argument de cette méthode.
   - Ce qui ne vaut que pendant un appel (un drapeau, un curseur, la forme en cours de lecture) est
     un argument ou une variable locale, jamais un attribut.
   - Exemple : `BatchReplacementService(reads, writes, scope, id_field)`, puis `replace(records)`.
2. **Pas d'instance d'une classe dont le constructeur ne prend rien.** Une instance existe parce
   qu'elle garde quelque chose : une configuration, des dépendances, des paramètres ou des valeurs.
   Une classe qui ne garderait rien ne s'instancie pas :
   - une logique sans état est une fonction du module ;
   - une façon de construire une instance est une méthode de classe de la classe construite
     (`Problem.from_files(path)`) ;
   - un travail qui a besoin d'un état pendant un appel devient une méthode de la classe qui
     l'appelle.
3. **Une classe a une seule responsabilité.** Une classe qui en mêle deux (lire des plans et écrire
   des règles, par exemple) est coupée en deux. Une méthode qui n'a rien à faire dans sa classe en
   sort.
4. **Un contrat s'écrit en `ABC`, jamais en `Protocol`.**
   - Les implémentations héritent de l'`ABC` : le contrat est explicite et vérifié.
   - Une `ABC` a **plusieurs vraies sous-classes** et une raison de les réunir (un comportement
     commun, un appelant qui les prend toutes). Un faux d'essai ne compte pas comme sous-classe.
   - Une seule implémentation : pas d'abstraction.
   - Les patrons connus s'appliquent sous leur nom (Repository, Gateway, Strategy, Adapter) plutôt
     qu'une construction maison.
5. **Une valeur se vérifie une fois, quand l'objet est construit.**
   - Par un `model_validator` Pydantic, ou par le constructeur d'une classe simple.
   - Un enregistrement vérifié est de préférence figé (`frozen=True`). S'il doit changer,
     `validate_assignment=True` revérifie chaque modification.
   - Une propriété rend une valeur ; elle ne revérifie pas un type à chaque lecture.
6. **Plutôt refuser que mentir.** Le code ne rend jamais un résultat faux en silence.
   - Ce qu'il ne sait pas traiter exactement est refusé au plus tôt, par une exception claire.
   - Une approximation n'est acceptée que si elle reste vraie : une borne plus faible mais toujours
     valide, par exemple.

## Nécessité

7. **Tout ce qui existe sert.** Chaque module, classe, méthode, champ, paramètre et exception est
   nécessaire ; ce qui n'est ni utilisé ni testé est retiré. Une enveloppe sans comportement ni
   information propre est de la sur-ingénierie. Par exemple, une exception maison vide
   (`UnsupportedError`) cède la place à une exception standard (`ValueError`).
8. **Une seule façon de faire chaque chose.**
   - Pas de `@overload`, pas de signature « une chose ou une liste », pas de deux méthodes publiques
     pour le même effet.
   - Pas de deuxième implémentation à côté de celle qui existe (une deuxième recherche, un deuxième
     chargement) : on appelle l'existante.
9. **Ni variable globale ni enveloppe triviale.**
   - Les constantes vivent dans la classe qui s'en sert.
   - Pas de lambda ni de fonction d'une ligne qui ne fait que relayer un appel ou renommer une
     expression.
   - Seule exception : une lambda qui choisit un champ du schéma dans une déclaration
     (`model.param(lambda op: op.duration, operations)`), pour que l'IDE vérifie le chemin.

## Organisation

10. **Une classe vit à côté du code qui s'en sert,** dans son fichier ou son dossier. Pas de dossier
    fourre-tout (`schema/`, `utils/`), pas de dossier propre à un client ou à un problème.
11. **Le code de `core/` est générique.** Il ne nomme ni un client, ni un problème, ni une entité
    métier. Un besoin propre à un problème passe par un paramètre ou une abstraction générique,
    jamais par du code qui nomme ce problème. Ce qui est propre à un problème ne sert qu'aux essais.
12. **Aucun nom de classe ne commence par un souligné.** Une classe intermédiaire, comme l'état d'une
    exécution, vit dans le fichier de la classe qui s'en sert, sous un nom normal (`PrecheckRun`, pas
    `_Run`).
13. **Le code ne sait pas qu'il est testé.** Rien dans `src/` n'existe pour les essais : ni réglage, ni
    variable d'environnement, ni branche. Un essai lent se corrige dans `tests/`. Une dépendance
    injectée (un transport HTTP, une horloge) n'est pas un réglage d'essai : la production s'en sert
    aussi.
14. **Les essais suivent l'arborescence du code.** L'essai de `src/baseline_core/<chemin>/<module>.py`
    est `tests/<chemin>/<module>_test.py`. Les modèles et bancs d'essai d'un paquet vivent dans son
    dossier d'essais. Un essai qui porte sur tout le dépôt (`architecture_test.py`) vit à la racine de
    `tests/`.
15. **Un script est un point d'entrée court qui montre ce que les essais ne montrent pas** : une
    mesure, un banc, un usage complet. Ses classes vivent dans le paquet ou dans `tests/`. Un script
    qui ne montre rien de neuf est retiré.

## Écriture

16. **Une méthode sans argument devient une propriété seulement si sa lecture est sans surprise.**
    - Une propriété se lit comme un attribut : le lecteur suppose qu'elle est immédiate, sans effet et
      qu'on peut la lire dix fois sans coût.
    - Elle reste une méthode dès qu'un doute existe : un calcul coûteux, une entrée-sortie (fichier,
      réseau, base), un effet de bord, un résultat qui change d'un appel à l'autre, ou un `async`. Les
      parenthèses de l'appel montrent alors qu'il y a un travail.
    - Exemple : `task.end` (début + durée) est une propriété ; `schedule.render()` (coûteux) et
      `clock.now()` (change à chaque appel) restent des méthodes.
17. **Plus de quatre arguments : se demander s'il y en a trop.** *(signal)* `self` non compté,
    constructeur compris. Souvent, des arguments qui vont ensemble forment un enregistrement, ou la
    méthode fait trop de choses et se coupe. Si chacun est une donnée distincte et nécessaire, ils
    restent.
18. **Des listes, pas des tuples.** Un tuple demande une raison : une valeur qui doit être immuable
    ou hachable (une clé de dictionnaire, un membre d'ensemble). Sinon une liste, un dictionnaire ou
    un enregistrement.
19. **Les commentaires sont rares et courts** : une ou deux lignes qui disent une contrainte que le
    code ne montre pas. Aucun séparateur de section.
20. **Une docstring tient en une ligne,** plus une ligne par entrée `Args`, `Returns` et `Raises`. Dans
    une docstring ou un document, une méthode s'écrit avec ses arguments (`run(bound)`) ou sans
    parenthèses (`run`), jamais `run()`.
21. **Le format est fixe** : lignes de 99 caractères au plus, code et noms en anglais, annotations de
    type partout.
22. **Le code respire.**
    - Une ligne vide avant et après chaque bloc (`if`, `for`, `while`, `with`, `try`), sauf au début
      ou à la fin du bloc qui le contient.
    - Une ligne vide entre les étapes d'une méthode.
    - Un essai sépare préparation, action et vérification par une ligne vide.

## Erreurs et journaux

23. **Une exception remonte jusqu'au code qui sait la traiter.**
    - Le code lève une exception précise. Il ne l'attrape que pour s'en remettre ou pour la traduire.
    - Pas de `except` qui avale l'erreur, pas de `try` qui journalise puis relance.
    - La traduction en réponse HTTP se fait une seule fois, dans `ExceptionHandlerMiddleware`. Un
      service ou un repository ne connaît pas HTTP.
    - Un `except Exception` n'existe qu'à une frontière finale (une requête, un processus enfant), sa
      raison écrite dans son `noqa`.
24. **On journalise avec `loguru`, jamais avec `print`.** Une erreur est journalisée une seule fois, là
    où elle est traitée (`logger.exception`). Aucun secret ni donnée personnelle dans un journal.
