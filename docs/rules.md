# Règles de revue

Ces règles servent de grille à toute revue de code de baseline-core. Une revue cite leur numéro.
Un module qui a des règles propres les porte dans sa documentation : celles du moteur
d'optimisation sont dans `docs/optimization-engine.md` (« Règles du moteur »).

## Conception

1. **Constructeur = configuration, méthodes = données.** Le constructeur reçoit ce qui est donné une
   fois ; une méthode reçoit les données qu'elle traite. Un argument propre à une seule méthode y
   reste. Ce qui ne vaut que pendant un appel (un drapeau, un curseur, la forme en cours de lecture)
   est un argument ou une variable locale, jamais un attribut. Les enregistrements reçoivent leurs
   valeurs au constructeur. Exemples : `InfeasibilityCheck(config)` puis `run(bound)` ;
   `BatchReplacementService(reads, writes, scope, id_field)` puis `replace(records)`.
2. **Pas de classe sans configuration.** Une classe dont le constructeur n'aurait aucun argument
   n'existe pas. Une logique sans état est une fonction du module ; une façon de construire une
   instance est une méthode de classe de la classe construite (`Problem.from_files(path)`) ; un état
   de travail va dans la classe configurée qui s'en sert.
3. **Une classe, une responsabilité.** Une classe qui en mêle deux (lire des plans et écrire des
   règles, par exemple) est coupée. Une méthode qui n'a rien à faire dans sa classe en sort.
4. **Des interfaces fortes : une `ABC`, jamais un `Protocol`.** Un contrat s'écrit en `ABC` et ses
   implémentations en héritent. Une `ABC` a **plusieurs vraies sous-classes** et une raison de les
   réunir (un comportement commun, un appelant qui les prend toutes) ; un faux d'essai ne compte pas
   comme sous-classe. Pas d'abstraction pour une seule implémentation. On applique les patrons connus
   sous leur nom (Repository, Gateway, Strategy, Adapter) plutôt qu'une construction maison.
5. **Valider à la construction, pas à la lecture.** Une valeur est vérifiée une fois, quand l'objet
   est fait : par un `model_validator` Pydantic, ou par le constructeur d'une classe simple. Un
   enregistrement validé est figé (`frozen=True`), sinon la validation ne tient plus. Une propriété
   rend une valeur ; elle ne vérifie pas un type pour lever une erreur à chaque lecture.
6. **Sûr, sinon refusé.** Le code ne rend jamais un résultat faux en silence. Ce qu'il ne sait pas
   traiter exactement est refusé au plus tôt, par une exception claire. Une approximation n'est
   acceptée que si elle reste vraie (une borne plus faible, mais vraie).

## Nécessité

7. **Tout doit servir.** Chaque module, classe, méthode, champ, paramètre et exception est
   nécessaire ; ce qui n'est ni utilisé ni testé est retiré. Du code qui n'apporte rien est de la
   sur-ingénierie : une enveloppe sans comportement ni information propre est retirée. Par exemple,
   une exception maison vide (`UnsupportedError`) cède la place à une exception standard
   (`ValueError`).
8. **Une seule façon de faire chaque chose.** Pas de `@overload`, pas de signature « une chose ou une
   liste », pas de deux méthodes publiques pour le même effet. Pas de deuxième implémentation à côté
   de celle qui existe (une deuxième recherche, un deuxième chargement) : on appelle l'existante.
9. **Ni variable globale ni enveloppe triviale.** Les constantes vivent dans leur classe. Pas de
   lambda ni de fonction d'une ligne qui ne fait que relayer ou renommer une expression. Seule
   exception : une lambda qui choisit un champ du schéma dans une déclaration
   (`model.param(lambda op: op.duration, operations)`), pour que l'IDE vérifie le chemin.

## Organisation

10. **Une classe vit là où elle sert.** Pas de dossier fourre-tout (`schema/`, `utils/`), pas de
    dossier propre à un client ou à un problème.
11. **Générique, sinon c'est un échec.** Rien dans `core/` ne nomme un client, un problème ou une
    entité métier. Un besoin propre à un problème passe par une primitive générique déclarée, jamais
    par du code qui le nomme. Ce qui est propre à un problème ne sert qu'aux essais.
12. **Pas de classe préfixée d'un souligné.** Une représentation intermédiaire, comme l'état d'une
    exécution, vit dans le fichier de la classe qui s'en sert, sous un nom normal (`PrecheckRun`, pas
    `_Run`). Elle n'a pas de fichier à elle.
13. **Le code ignore les essais.** Rien dans `src/` n'existe pour les essais : ni réglage, ni variable
    d'environnement, ni branche. Un essai lent se corrige dans `tests/`.
14. **Les essais suivent l'arborescence du code.** L'essai de `src/baseline_core/<chemin>/<module>.py`
    est `tests/<chemin>/<module>_test.py` ; les modèles et bancs d'essai d'un paquet vivent dans son
    dossier d'essais.
15. **Un script est un point d'entrée court qui montre ce que les essais ne montrent pas** (une
    mesure, un banc, un usage complet). Ses classes vivent dans le paquet ou dans `tests/` ; un script
    qui ne montre rien de neuf est retiré.

## Écriture

16. **Méthode sans argument qui rend une valeur → propriété,** quand elle est bon marché, sans
    entrée-sortie et non `async`.
17. **Au plus quatre arguments par méthode,** `self` non compté, constructeur compris. Au-delà, les
    arguments qui vont ensemble forment un enregistrement, ou la méthode fait trop de choses et se
    coupe.
18. **Des listes, pas des tuples.** Un tuple demande une vraie raison : une valeur qui doit être
    immuable ou hachable (une clé de dictionnaire, un membre d'ensemble). Sinon une liste, un
    dictionnaire ou un enregistrement.
19. **Commentaires rares et courts** : une ou deux lignes, une contrainte que le code ne montre pas ;
    aucun séparateur de section.
20. **Docstrings d'une ligne,** une ligne par entrée `Args`, `Returns` et `Raises`. Dans un texte, une
    méthode s'écrit avec ses arguments (`run(bound)`) ou sans parenthèses (`run`), jamais `run()`.
21. **Format** : lignes de 99 caractères au plus, code et noms en anglais, annotations de type
    partout.
22. **Du code qui respire.** Une ligne vide avant et après chaque bloc (`if`, `for`, `while`, `with`,
    `try`), sauf quand il ouvre ou ferme le bloc qui le contient ; une ligne vide entre les étapes
    d'une méthode. Un essai sépare préparation, action et vérification par une ligne vide.

## Erreurs et journaux

23. **Une exception remonte jusqu'à qui sait la traiter.** Le code lève une exception précise et ne
    l'attrape que pour s'en remettre ou la traduire. Pas de `except` qui avale, pas de `try` qui
    journalise puis relance. La traduction en réponse HTTP se fait une seule fois, dans
    `ExceptionHandlerMiddleware` : un service ou un repository ne connaît pas HTTP. Un
    `except Exception` n'existe qu'à une frontière finale (une requête, un processus enfant), sa
    raison écrite dans son `noqa`.
24. **Journaliser avec `loguru`,** jamais `print`. Une erreur est journalisée une fois, là où elle est
    traitée (`logger.exception`). Aucun secret ni donnée personnelle dans un journal.

## Changer du code

25. **Même résultat après une optimisation ou une refactorisation.** Les sorties restent identiques,
    et les essais le vérifient. Un écart n'est accepté que s'il corrige un défaut, et il est alors
    nommé.
26. **Rien n'est retiré avant d'avoir vérifié qu'il ne sert pas** : à un contrôle qu'il reproduit, à
    une preuve, à un essai. La justification est écrite, et le retrait est un commit à part, qu'on
    peut annuler.
27. **Une référence à un outil extérieur est vraie.** Une docstring ou un document ne cite une API,
    une primitive ou un format d'un outil extérieur qu'après avoir vérifié, dans sa documentation,
    qu'il existe et qu'il a ce sens.
28. **Le code reçu d'ailleurs passe ces règles avant d'être installé.** Un paquet livré dans un zip ou
    par une autre session est relu classe par classe, script par script ; ce qui ne passe pas est
    réécrit ou retiré, jamais seulement déplacé.
29. **Le parallélisme n'utilise jamais tous les cœurs** : au plus les cœurs disponibles moins deux, au
    moins un, et seulement si un gain mesuré le justifie.
30. **Revue.** Un agent (`@dev`) fait une première revue contre ces règles ; une approbation humaine
    reste requise avant la fusion. L'auteur et les réviseurs doivent s'accorder, chacun ne cédant que
    devant un vrai argument. Une meilleure option est bienvenue si elle est justifiée.
