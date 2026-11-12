# Règles de revue

Ces règles servent de grille à toute revue de code. Une revue cite leur numéro. Les exemples viennent
du moteur d'optimisation (`src/baseline_core/core/optimization/`, voir `docs/optimization-engine.md`).

1. **Constructeur = configuration, méthodes = données.** Le constructeur reçoit ce qui est donné une
   fois ; une méthode reçoit les données qu'elle traite et peut changer l'état de l'objet. Un argument
   propre à une seule méthode y reste (`seed` de `min_conflicts`). Un drapeau qui
   ne vaut que pendant un appel est un argument, pas un attribut. Les enregistrements (nœuds,
   déclarations, `Term`, `PrecheckIssue`, `TermShape`, `Template`) reçoivent leurs valeurs au constructeur.
   Exemples : `ModelCompiler(definition, cythonize=)` puis `bind(problem)` ;
   `Binder(model)` puis `expand(data)` ; `InfeasibilityCheck(config)` puis `run(bound)`.
2. **Pas de classe sans configuration instanciée** (règle, pas une option). Une classe dont le
   constructeur n'aurait aucun argument ne s'instancie pas : elle n'a que des méthodes de classe ou
   statiques, comme une fabrique qui assemble une instance de problème à partir de fichiers. Si elle a
   besoin d'un état de travail pendant un appel, elle est fondue dans la classe configurée qui s'en sert
   (pas de `cls()` sur un constructeur vide).
3. **Méthode sans argument qui rend une valeur → propriété** quand c'est pertinent.
4. **Une seule façon d'écrire chaque chose** : pas de `@overload`, pas de signature « une chose ou une
   liste », pas de deux méthodes publiques pour le même effet.
5. **Pas de `Protocol`** (une `ABC` au besoin) ; pas d'abstraction pour une seule implémentation.
   Une `ABC` a **plusieurs vraies sous-classes** et une raison de les réunir (un comportement commun,
   un appelant qui les prend toutes) ; un faux d'essai ne compte pas comme sous-classe : un encodeur
   abstrait dont la seule sous-classe sert un banc d'essai est retiré.
6. **Ni variable globale ni fonction d'une ligne** (lambda, aide triviale) ; les constantes vivent dans
   leur classe. Seule exception : une lambda qui choisit un champ du schéma dans une déclaration
   (`model.param(lambda op: op.duration, operations)`), pour que l'IDE vérifie le chemin.
7. **Pas de classe inutile, pas de sur-ingénierie, pas de dette technique, pas de méthode qui n'a rien
   à faire dans sa classe** ; ce qui n'est ni utilisé ni testé est retiré. **Du code qui n'apporte rien
   est de la sur-ingénierie.** Une classe, une exception, un champ, un paramètre ou une enveloppe sans
   comportement ni information propre est retiré. Par exemple, une exception maison vide
   (`UnsupportedError`) cède la place à une exception standard (`ValueError`).
8. **Organisation** : le moteur garde ses dossiers (`problem_formulation`, `solver`, `mining`) ; une
   classe vit là où elle sert ; pas de dossier `schema`, pas de dossier propre à un problème, pas
   d'OR-Tools, pas d'encodeur/décodeur (`SchemaPath` en tient lieu).
9. **Générique, sinon c'est un échec** : rien dans `optimization/` ne nomme un problème. Le catalogue
   ne garde que des primitives des outils de référence. Un besoin propre à un problème (une pause de
   midi, un temps de nettoyage entre deux sortes de tâches) passe par une primitive générique déclarée
   (`model.breaks`, `Transition`), jamais par du code qui le nomme. Les modèles propres à un problème
   ne servent qu'aux essais.
10. **Pas d'interpréteur** : le noyau est en Cython « mode Python pur », le même code compilé ou non.
11. **Pré-vérification** : trois niveaux, **le délai réglé par l'appelant, sans défaut** ; ce qui n'est
    pas décidé à temps n'est pas prouvé. Si un cas n'est pas tenu, on le dit : aucun essai ni réglage
    ajusté pour masquer une lenteur. Non générique pour n'importe quel problème = échec.
12. **Commentaires rares et courts** (un ou deux lignes, une contrainte que le code ne montre pas), aucun
    séparateur de section ; une méthode qui prend des arguments s'écrit avec eux (`run(bound)`), ou
    sans parenthèses (`run`), jamais `run()` ; docstrings d'une ligne, une ligne par entrée
    `Args`/`Returns` ; lignes ≤ 99 caractères ; code et noms en anglais ; annotations de type partout.
13. **Revue** : un agent indépendant (`@dev`) vérifie ces règles contre le code ; l'auteur et les
    réviseurs doivent s'accorder, chacun ne cédant que devant un vrai argument. Une meilleure option est
    bienvenue si elle est justifiée.
14. **Chaque module, classe et méthode doit être nécessaire.** Une revue se demande pour chacun s'il sert
    vraiment (un interpréteur d'expressions, par exemple, ne sert pas quand le noyau tourne aussi en
    Python pur). Pour s'inspirer : `solver/kernel/kernel.py` (noyau Cython en mode Python pur, le même
    fichier compilé ou non) et le découpage de `solver/` (`binding/`, `kernel/`, `infeasibility/`,
    `search/`). **Une classe, une responsabilité** : une classe qui en mêle deux (lire les plans et
    écrire les règles, par exemple) est coupée ; une classe d'une seule méthode sans état est une
    fonction. **Une seule façon de faire chaque chose** : une deuxième recherche, un deuxième chargement
    du noyau, une deuxième lecture des durées à côté de celles du moteur sont retirés au profit du
    moteur (un échantillonneur de plans appelle `KernelSolver`, il ne refait pas sa boucle).

15. **Valider à la construction, pas à la lecture.** Une valeur est vérifiée une fois, quand l'objet est
    fait : par un `model_validator` Pydantic, ou par le constructeur d'une classe simple. Une propriété
    rend une valeur ; elle ne vérifie pas un type pour lever une erreur à chaque lecture (par exemple
    une propriété `mask` qui vérifierait, à chaque lecture, qu'un masque est déjà un tableau NumPy).
16. **Sûr, sinon refusé.** Le moteur ne rend jamais un résultat faux en silence : une preuve
    (`PrecheckIssue` prouvé) doit être vraie, et une variation doit égaler un recompte complet. Ce que le
    moteur ne sait pas traiter exactement est refusé par une exception claire à la liaison (par exemple
    un élément nommé par des données, `at[partner[e]]`). Une borne qui ne tient pas dans un cas (une
    pause qui chevauche deux périodes) retombe sur une borne plus faible mais vraie.
17. **Même résultat après une optimisation ou une refactorisation.** Comptes, variations, violations et
    constats restent identiques. C'est vérifié par script contre la version d'avant, compilé et en
    Python pur, sur plusieurs modèles d'essai (atelier, chargement de camions, ressources). Un écart
    n'est accepté que s'il corrige un défaut, et il est alors nommé.
18. **Rien n'est retiré avant d'avoir vérifié qu'il ne sert pas** : à un contrôle métier qu'il
    reproduit, à une preuve, à un essai. La justification est écrite. Ce qui est retiré est copié de
    côté d'abord.
19. **Une référence à un outil extérieur est vraie.** Une docstring ou un document ne cite une primitive
    (CP-SAT, CP Optimizer, OR-Tools, MiniZinc) qu'après avoir vérifié qu'elle existe et qu'elle a ce
    sens. `AddBinPacking` de CP-SAT, qui n'existe pas, a été cité à tort.
20. **Pas d'état de travail sur l'objet** : ce qui ne vaut que pendant un appel (la forme en cours de
    lecture, un drapeau, un curseur) est un argument ou une variable locale, pas un attribut.
21. **Le parallélisme n'utilise jamais tous les cœurs** : au plus `available_cores() - 2`, et seulement si
    un gain mesuré le justifie.

22. **Pas de classe préfixée d'un souligné.** Une représentation intermédiaire, comme l'état d'une
    exécution (`_Run`), vit dans le fichier de la classe qui s'en sert, sous un nom normal
    (`PrecheckRun`). Elle n'a pas de fichier à elle : elle fait partie de ce fichier.

23. **Au plus quatre arguments par méthode.** Au-delà, les arguments qui vont ensemble forment un
    enregistrement (règle 22), ou la méthode fait trop de choses et se coupe.

24. **Des listes, pas des tuples.** Un tuple rendu ou reçu demande une vraie raison ; sinon une liste,
    un dictionnaire ou un enregistrement.

25. **Le moteur ignore les essais.** Rien dans `optimization/` n'existe pour les essais : ni réglage, ni
    variable d'environnement, ni branche (par exemple un `-O0` pour compiler plus vite). Un essai lent
    se corrige dans `tests/` : Python pur par défaut, Cython seulement pour comparer les deux.

26. **Le code reçu d'ailleurs passe ces règles avant d'être installé.** Un paquet livré dans un zip ou
    par une autre session est relu classe par classe, script par script ; ce qui ne passe pas est
    réécrit ou retiré, jamais seulement déplacé. Un paquet qui porte une deuxième recherche, une `ABC`
    à une seule implémentation ou des classes de banc dans `scripts/` ne s'installe pas tel quel.

27. **Un script est un point d'entrée court qui montre ce que les essais ne montrent pas** (une mesure,
    un banc, un usage complet). Ses classes vivent dans le paquet ou dans `tests/` (un banc est un
    modèle d'essai, comme `tests/core/optimization/job_shop_model.py`) ; un script qui ne montre rien
    de neuf est retiré.

28. **Du code qui respire.** Une ligne vide avant et après chaque bloc (`if`, `for`, `while`, `with`,
    `try`), sauf quand il ouvre ou ferme le bloc qui le contient ; une ligne vide entre les étapes d'une
    méthode. Un essai sépare préparation, action et vérification par une ligne vide.

29. **Les essais suivent l'arborescence du code.** L'essai de
    `src/baseline_core/core/optimization/<paquet>/<module>.py` est
    `tests/core/optimization/<paquet>/<module>_test.py` ; les modèles et bancs d'essai d'un paquet
    vivent dans son dossier d'essais (`tests/core/optimization/mining/shop.py`).
