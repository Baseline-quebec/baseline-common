# Règles de revue

Ces règles servent de grille à toute revue de code de baseline-common et des dépôts spécialisés
(baseline-optimization, baseline-ml, baseline-llm), qui n'en gardent pas de copie. Une revue cite
leur numéro. Un dépôt spécialisé a les couches d'ici ; tout son code générique tient dans un seul
paquet sous `framework/`, qu'on importe par son chemin complet : `baseline_ml/framework/ml/`,
`baseline_llm/framework/llm/`, `baseline_optimization/framework/optimization/`.
Chaque règle tient dans son titre en gras ; le texte qui suit l'explique et donne un exemple.
Une règle marquée *(signal)* n'interdit rien : elle pose une question à la revue, et un écart
justifié est accepté.

La revue vérifie ces règles ; ruff en vérifie une partie, là où le texte nomme son code (BLE001).
Les règles de structure (couches, points d'entrée, assembly, imports entre dépôts, nommage des
fichiers) sont dans le `README.md`, sous « Définitions », « Règles d'architecture » et
« Conventions de nommage » ; « Règles d'architecture » dit ce que l'essai d'architecture vérifie.

Un module qui a des règles propres les porte dans sa documentation : celles du moteur
d'optimisation sont dans `docs/optimization-engine.md` de baseline-optimization (« Règles du moteur »).

## Conception

1. **Le constructeur reçoit ce qui configure l'objet, les méthodes reçoivent les données.**
   - Le constructeur reçoit ce qui est donné une fois : réglages, dépendances, paramètres.
   - Une méthode reçoit les données qu'elle traite. Un argument qui ne sert qu'à une méthode reste
     un argument de cette méthode.
   - Ce qui ne vaut que pendant un appel (un drapeau, un curseur, la forme en cours de lecture) est
     un argument ou une variable locale, jamais un attribut.
   - Exemple : `SqlRepository(session, Machine, scope=scope)`, puis `insert_many(records)`.
   - Ce qui entre dans une méthode d'un service est l'une de ces quatre choses, et rien d'autre :
     - un enregistrement de `models/` : `RecordMatchingService.match(request)` reçoit un
       `MatchRequest` ;
     - un enregistrement d'un module de `framework/` (une affectation que rend le moteur) ;
     - une donnée brute, prise telle quelle : des octets, une image, un tableau ;
     - une valeur qui se suffit à elle-même : un identifiant, un nom à normaliser, un membre d'une
       énumération, un drapeau, la période qu'une route lit dans son chemin.

     Des valeurs qui décrivent ensemble une même chose forment un enregistrement de `models/`,
     jamais une suite d'arguments `int` ou `str`.
   - Une donnée de la requête, un paramètre de chemin compris, est une donnée : la route la passe
     à la méthode. Un provider ne lit jamais la requête pour la donner à un constructeur.
   - Un agent pydantic-ai suit la même règle : le constructeur le configure (modèle, instructions),
     sa méthode reçoit les données dans un objet requête. Son `deps` ne porte que ce dont ses
     outils ont besoin pendant l'exécution ; sans outil, c'est `None` (`Agent[None, MatchDecision]`).
2. **On instancie une classe pour la configurer.** Les arguments du constructeur (réglages,
   dépendances, paramètres) sont ce qui configure l'objet. Une classe dont le constructeur ne
   prendrait aucun argument n'a rien à configurer, donc ne s'instancie pas.
   - Une logique sans état est une méthode statique d'une classe qu'on n'instancie jamais ; cette
     classe porte aussi les constantes de cette logique. On l'appelle sur la classe :
     `Imports.names(code)` dans l'essai d'architecture, jamais `Imports().names(code)`.
   - Une façon de construire une instance est une méthode de classe de la classe construite
     (`Problem.from_files(path)`).
   - Un travail qui a besoin d'un état pendant un appel devient une méthode de la classe qui
     l'appelle.
   - Une famille d'`ABC` se décide en entier, d'après ses implémentations :
     - aucune ne reçoit de réglage : rien ne s'instancie. Les méthodes abstraites sont
       `@classmethod` et `@abstractmethod`, et l'appelant reçoit la classe
       (`codec: type[AbstractCodec]`, puis `codec.encode(boxes)`) ;
     - au moins une reçoit des réglages : toutes s'instancient, même celles qui ne reçoivent rien,
       pour que l'appelant les traite de la même façon. La docstring de l'`ABC` dit pourquoi :
       laquelle reçoit des réglages, et lesquels.
   - Jamais de fonction de module. Seules exceptions :
     - ce qu'un point d'entrée (`api/`, `cli/`, le `worker/` d'un projet) déclare au niveau du
       module : son `init_container()`, son `get_settings()`, la fonction qu'il lance dans un
       processus enfant, les commandes d'une ligne de commande, et ce que son framework veut au
       niveau du module (les routes, `init_app(container)` et les lifespans de FastAPI) ;
     - dans les essais, les fixtures et les hooks pytest (`pytest_configure`).
3. **Une classe a une seule responsabilité.** Une classe qui en mêle deux (lire des plans et écrire
   des règles, par exemple) est coupée en deux. Une méthode qui n'a rien à faire dans sa classe en
   sort.
4. **Un contrat s'écrit en `ABC`.**
   - Les implémentations héritent de l'`ABC` : le contrat est explicite, et un oubli est signalé dès
     l'instanciation.
   - Une `ABC` a **plusieurs vraies sous-classes** et une raison de les réunir (un comportement
     commun, un appelant qui les prend toutes). Un faux d'essai ne compte pas comme sous-classe.
   - Une seule implémentation, ou aucune hors des essais : pas d'abstraction. Seule exception : un
     point d'extension où d'autres implémentations sont raisonnablement attendues (un autre
     backbone, ou celle que chaque projet écrit) ; la docstring de l'`ABC` écrit cette attente.
     Sans elle, l'`ABC` est retirée et l'appelant reçoit la classe concrète.
   - Le nom et le fichier d'une `ABC` (`abstract_<nom>.py`, ou le nom nu d'une famille définie dans
     un seul fichier) suivent les « Conventions de nommage » du `README.md`.
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
   - Seules exceptions aux variables globales :
     - les objets qu'un point d'entrée déclare au niveau du module parce que son framework le
       veut : `app` et `container` dans `api.py`, `router` dans un fichier de routeur,
       l'application d'une ligne de commande (`app = typer.Typer()`). Les réglages n'en font pas
       partie : chaque usage appelle `get_settings()`, mis en cache, plutôt que de lire une
       variable `settings` du module ;
     - le `__all__` d'un module qui liste ses noms publics, quand un outil les lit. Jamais dans un
       `__init__.py` (« Conventions de nommage » du `README.md`).
   - Pas de lambda ni de fonction d'une ligne qui ne fait que relayer un appel ou renommer une
     expression.
   - Seule exception : une lambda qui choisit un champ du schéma dans une déclaration
     (`model.param(lambda op: op.duration, operations)`), pour que l'IDE vérifie le chemin.
   - Un alias de type au niveau du module est permis quand il est justifié, et seulement alors :
     il nomme un type composé de plusieurs types (une union, la signature d'un `Callable`) qui sert
     à plusieurs endroits, ou c'est un `NewType` qu'un conteneur injecte par son type. Un
     commentaire au-dessus de lui dit pourquoi ; pas de docstring sous lui, le hook
     `check-docstring-first` la refuse. Un alias qui renomme une seule classe, même paramétrée
     (`Object[dict[str, Any], None]`), ou qui ne sert qu'une fois, est retiré :

     ```python
     # A type of its own, so a tenant can be injected by type.
     TenantID = NewType("TenantID", str)
     ```

## Organisation

9. **Le code de `framework/` est générique.** Il ne nomme ni un client, ni un problème, ni une entité
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
    - **Chaque attribut déclaré dans une classe est documenté** : champ, `ClassVar`, constante de
      classe, membre d'une énumération, `model_config`. Une ligne courte qui dit ce qu'il est,
      sans plus. Deux formes, et seulement elles :
      - un champ déclaré par `Field(...)` pour sa validation (`Field(ge=1, …)`), ou dont un schéma
        lit la description (OpenAPI, l'outil ou la sortie d'un agent), se documente par la
        `description` de ce `Field` ;
      - tout autre attribut, par une docstring d'une ligne placée juste sous lui. On n'écrit pas
        `Field(description=…)` pour la seule documentation.

      Jamais par une section `Attributes:` dans la docstring de la classe, ni par un commentaire
      au-dessus de lui. Un attribut que le constructeur pose (`self._session`) se documente par
      l'entrée `Args` du constructeur :

      ```python
      SCHEMA: ClassVar[str] = "v1"
      """Version of the schema the API serves."""
      limit: int = Field(ge=1, description="How many records a page holds at most.")
      ```
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
    servent. `tests/conftest.py` ne garde que ce qui sert partout. Une fixture ne remonte pas « au
    cas où ».
    - Plusieurs fichiers s'en servent : le `conftest.py` du plus petit dossier qui les contient tous.
    - Un seul fichier s'en sert : ce fichier.
    - Une seule classe d'essais s'en sert : ce fichier, ou cette classe ; les deux sont permis.
20. **Un essai respecte les couches du code qu'il teste.** Ses imports suivent le même sens que le
    code : `api, cli → assembly → orchestrators → services → models → framework`.
    - Un essai sous `tests/framework/` n'importe ni `models/`, ni `services/`, ni `orchestrators/`,
      ni `assembly/`, ni `api/`, ni les essais de ces couches.
    - Un essai d'un module de `framework/` n'importe pas un autre module de `framework/`.
    - Entre dépôts, le sens est celui des « Règles d'architecture » du `README.md` : un dépôt
      spécialisé lit `framework/` ou `models/` de baseline-common, son `framework/` seulement
      `framework/` (`framework/ml/graph/` de baseline-ml lit `framework/multidigraph/`).
21. **Un essai se nomme `test__<sujet>__<résultat>`.**
    - Le sujet dit ce que l'essai met en place, le résultat ce qu'il vérifie :
      `test__a_blank_name__is_refused`.
    - Une classe d'essais se nomme `Test<Classe>` quand elle regroupe les essais d'une classe
      (`TestRecordMatchingService`), ou `Test<Thème>` quand elle regroupe ceux d'un thème
      (`TestLayers` dans l'essai d'architecture). Les deux sont permis.
    - Ce format prime sur `test__when_<condition>__<result>` du guide du développeur.
    - Pas d'essai BDD (pytest-bdd, Gherkin) dans baseline-common : un framework n'a pas de user
      story. Un projet client qui en a les écrit dans ses `tests/features/`.

## Erreurs et journaux

22. **Une exception remonte jusqu'au code qui sait la traiter.**
    - Le code lève une exception précise. Il ne l'attrape que pour s'en remettre ou pour la traduire.
    - Une traduction lève une exception précise dont le message dit la raison, et garde la cause
      par `from` (ruff, EM101 et EM102, veut le message dans une variable) :

      ```python
      except ValidationError as error:
          msg = f"{source}: unreadable row, {field}: {detail}"
          raise UnreadableAnswerError(msg) from error
      ```
    - Un `except` qui ne relance pas s'en remet : il attrape une exception précise
      (`except ValueError: body = None`).
    - Seule une frontière finale attrape `Exception` pour l'avaler. Il y en a quatre, et aucune
      autre : une requête, un processus enfant, la dernière garde d'un fil d'exécution (thread), et
      un cas d'un lot dont l'échec est consigné sans arrêter les autres cas. Elle porte
      `# noqa: BLE001 - <raison>`. Dans un processus enfant lancé par `fork`, elle peut attraper
      `BaseException` : rien ne doit remonter dans la copie du parent.
    - Ailleurs qu'à une frontière finale, `except Exception` n'apparaît pas, même pour traduire : on
      attrape l'exception précise que le code appelé lève.
    - Ce que ruff signale : BLE001 marque tout `except Exception` qui ne relance pas, même s'il
      journalise avec loguru (`logger.exception`) : ruff ne reconnaît que le module `logging`. Un
      `except` qui relance (`raise`, `raise … from err`) n'est pas marqué, et un `noqa: BLE001` posé
      sur lui est signalé inutile (RUF100) : sa raison est le message de l'exception levée.
    - Pas de `try` qui journalise puis relance. Un nettoyage va dans `finally`, jamais dans un
      `except` qui relance.
    - La traduction en réponse HTTP se fait une seule fois, dans `ExceptionHandlerMiddleware`
      (`RecordNotFoundError` en 404, `DuplicateRecordError` en 409). Un service ou un repository ne
      connaît pas HTTP.
23. **On journalise avec `loguru`, jamais avec `print`.** Une erreur est journalisée une seule fois, là
    où elle est traitée (`logger.exception`). Aucun secret ni donnée personnelle dans un journal.
