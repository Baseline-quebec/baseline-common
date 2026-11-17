# baseline-common

## Raison d'être

baseline-common est la base de connaissances commune des développeurs de Baseline. On y trouve le
code qui a fait ses preuves dans un projet et qui servira au suivant :
- des briques génériques, utilisables chacune seule (`framework/`) ;
- les enregistrements du domaine que les couches s'échangent (`models/`) ;
- la façon de construire un service par-dessus `framework/`, et ses adapters (`services/`) ;
- la façon d'enchaîner des services en pipeline pour un processus (`orchestrators/`) ;
- la composition qui branche le domaine sur les points d'entrée (`assembly/`) ;
- les points d'entrée eux-mêmes (`api/`, et `cli/` dans un projet qui en a une).

Un projet généré par
[baseline-app-template](https://github.com/Baseline-quebec/baseline-app-template) choisit les
modules dont il a besoin, les copie, et en possède ensuite le code.

## Dépôts spécialisés

Trois modules ont quitté baseline-common pour leur propre dépôt. **Leur accès est réservé aux
administrateurs** de Baseline-quebec ; baseline-common reste ouvert à tous.

| Dépôt | Ce qu'il contient |
|---|---|
| [baseline-optimization](https://github.com/Baseline-quebec/baseline-optimization) | Le moteur d'optimisation : écrire un problème, le lier, le pré-vérifier, le résoudre ; ses règles propres. |
| [baseline-ml](https://github.com/Baseline-quebec/baseline-ml) | Entraîner, suivre, exporter et exécuter un modèle ; `framework/ml/graph/` lit `framework/multidigraph/` d'ici. |
| [baseline-llm](https://github.com/Baseline-quebec/baseline-llm) | L'extraction de documents, l'évaluation des sorties, le coût des jetons, et comment un agent se construit. |

Chacun a les couches d'ici. Tout son code générique tient dans un seul paquet sous `framework/`,
qu'on importe par son chemin complet : `baseline_ml/framework/ml/`, `baseline_llm/framework/llm/`,
`baseline_optimization/framework/optimization/`. Les modules de `framework/` ne s'importent pas
entre eux : avec un seul paquet, la règle tient d'elle-même, et l'essai d'architecture du dépôt la
garde.

Ils dépendent de baseline-common, jamais l'inverse. Les règles de revue générales
(`docs/rules.md`) ne vivent qu'ici ; chaque dépôt spécialisé y renvoie.

## Documentation

| Document | Contenu |
|---|---|
| [`docs/rules.md`](docs/rules.md) | **Les règles de revue du code,** communes à tout le dépôt. Tout code qui entre ici les respecte. |
| [`docs/architecture-services.md`](docs/architecture-services.md) | Les couches et comment un service, un orchestrateur et un repository se construisent. |
| [`docs/SOURCES.txt`](docs/SOURCES.txt) | La provenance de chaque fichier : dépôt, branche, commit, chemin. |

## Arborescence

```
docs/                            la documentation ci-dessus
docker/                          l'image des essais sur SQL Server (pilote ODBC de Microsoft)
iac/                             infrastructure as code (vide pour l'instant)
src/baseline_common/            de la couche la plus basse à la plus haute
├── framework/                   briques génériques, une librairie par tranche
│   ├── multidigraph/            graphe orienté multiple typé
│   └── repository/              lire ou écrire des enregistrements
│       ├── http/                une API HTTP distante (Zoho, Fidelio) ; client/, son client HTTP
│       ├── nosql/mongo/         MongoDB
│       ├── sql/                 tout dialecte SQLAlchemy async : PostgreSQL, SQL Server…
│       ├── vectorstore/         Qdrant, Weaviate
│       ├── abstract_repository.py
│       └── in_memory_repository.py
├── models/                      enregistrements du domaine partagés entre couches
├── services/                    un service par entité ou capacité
│   ├── record_matching/
│   │   ├── crm_adapter.py       lire un enregistrement du domaine dans la forme d'une source
│   │   └── record_matching_service.py  comment un service se construit
│   └── batch_replacement_service.py
├── orchestrators/               un pipeline par processus
├── assembly/                    la composition : les providers Dishka des services
└── api/                         point d'entrée FastAPI : api.py, api_version.py, settings.py,
                                 v1/ (api.py, lifespan.py, error_handling.py, routers/)
tests/                           même arborescence que src/ ; architecture_test.py
```

## Définitions

Dans l'ordre de l'arborescence.

- **Module du framework** : du code générique, sous `framework/`. Il ne nomme ni un client ni un
  problème. Le framework est une librairie par tranche : chaque module s'utilise et se copie seul,
  et porte ses classes abstraites (`abstract_<nom>.py`) avec leurs implémentations.
- **Repository** : lit **ou** écrit un seul type d'enregistrement, sans règle métier. Il est
  injecté dans un service, dans un orchestrateur quand l'étape n'est qu'une lecture ou une
  écriture, ou dans une route quand la requête n'est qu'une lecture (pas de service qui ne ferait
  que relayer).
- **Models** : les enregistrements Pydantic du domaine qu'une application partage entre ses
  couches (`api`, `orchestrators`, `services`) ou entre plusieurs services. C'est le vocabulaire
  commun de l'application, la couche juste au-dessus de `framework/`.
  - **Ce qui y entre :** un enregistrement lu ou produit par plus d'un service ou plus d'une
    couche (une entité du domaine, une requête ou une réponse que l'API expose et qu'un service
    produit). Un enregistrement qu'une seule classe utilise reste dans le fichier de cette classe :
    la sortie d'un agent à côté de lui, la configuration d'un service à côté de lui.
  - **Ce qu'il contient :** des champs, des validateurs d'invariants (un identifiant bien formé,
    une intégrité référentielle) et des propriétés dérivées simples. Aucune entrée-sortie, aucun
    appel de service, aucune règle métier lourde : une transformation de plusieurs centaines de
    lignes est un service, même si elle ne produit qu'un enregistrement.
  - **Ce qu'il importe :** `framework/` et Pydantic seulement. Il n'importe jamais `services/`,
    `orchestrators/`, `assembly/` ni `api/`, et `framework/` ne l'importe jamais (un module du
    framework reste copiable seul).
  - **Ici,** il porte les enregistrements de l'exemple de service : `Organization`, que lisent
    l'adapter, le service et l'API, puis `MatchRequest` et `MatchDecision`, l'entrée et la sortie
    du service, que l'API expose. Dans un projet, il porte ceux du domaine du client.
- **Service** : les opérations de base d'une entité ou d'une capacité, dont la conversion des
  données externes en données du domaine. Son constructeur reçoit la configuration et les
  dépendances, ses méthodes reçoivent les données. Un service a son dossier `services/<sujet>/`
  dès qu'un second fichier n'appartient qu'à lui (agent, solveur, config, adapters).
- **Adapter** : lit un enregistrement dans la forme d'un système externe et rend celui du domaine.
  Il est propre à sa source, donc vit dans le dossier du service qui s'en sert,
  `services/<sujet>/<source>_adapter.py`, jamais dans `framework/`. **Un adapter est
  optionnel :** il n'existe que si le service lit une source dont la forme diffère de celle du
  domaine. Un adapter a une structure précise, selon le cas.
  - **Le cas courant (90 à 95 % des cas) : un schéma de source à `AliasPath`.** Une classe Pydantic
    hérite de l'enregistrement du domaine et ne redéclare que la façon de lire chaque champ dans la
    source. Elle est donc un enregistrement du domaine : le service qui la reçoit ne connaît pas la
    source. Exemple : `services/record_matching/crm_adapter.py`.
    - **Renommer un champ** ne demande même pas d'adapter : l'alias du modèle que lit le repository
      suffit (`Field(alias="Account_Name")`).
    - **Un champ imbriqué** se lit par `AliasPath` :
      `Field(validation_alias=AliasPath("Account", "Legal_Name"))`. `AliasChoices` prend le premier
      chemin présent.
    - **Une valeur à convertir** passe par un `field_validator(mode="before")`, qui reçoit la valeur
      brute au bout du chemin. Une conversion que Pydantic offre déjà se règle dans `model_config`
      (`coerce_numbers_to_str=True` pour un identifiant numérique).
  - **Le cas rare : plusieurs sources pour un même enregistrement.** Une réservation de voyage
    réunit un vol, lu dans l'API d'une compagnie aérienne, et un séjour, lu dans celle d'un hôtel :

    ```
    services/booking/
    ├── airline_flight.py        AirlineFlight : le vol tel que l'API de la compagnie le répond
    ├── hotel_stay.py            HotelStay : le séjour tel que l'API de l'hôtel le répond
    ├── trip_adapter.py          TripAdapter : réunit un vol et un séjour en un Trip du domaine
    └── booking_service.py       reçoit des Trip ; ne connaît ni la compagnie ni l'hôtel
    ```

    - **Un schéma par source,** construit comme dans le cas courant (alias, `AliasPath`,
      validateurs), mais sur la forme de sa seule source : il n'hérite pas de l'enregistrement du
      domaine, dont il ne porte qu'une partie.
    - **Une classe d'adaptation,** `<Domaine>Adapter`, réunit les schémas en l'enregistrement du
      domaine (`Trip`, dans `models/`). Comme un service, son constructeur reçoit ce qui la
      configure (les repositories des deux sources) et sa méthode reçoit les données
      (`read(reference)` rend un `Trip`).
    - **Cette classe n'existe que si le cas arrive.** Tant qu'une seule source fournit
      l'enregistrement, le schéma à `AliasPath` suffit.
  - **Limite :** le gateway nomme les champs de l'API par l'`alias` du modèle. Un modèle à
    `AliasPath` se valide sur la réponse telle quelle (corps HTTP, fichier, message) ; il ne sert pas
    de `content_type` à un `ReadGateway`.
- **Orchestrateur** : le pipeline d'un processus. Il enchaîne plusieurs services, sans règle métier
  propre. C'est un seul fichier, `orchestrators/<processus>_orchestrator.py`, avec une ou deux
  méthodes publiques ; une de plus demande une raison (une autre entrée exposée à l'API). Ses
  méthodes privées ne comptent pas. C'est là que des modules du framework, qui s'ignorent entre
  eux, se rencontrent.
- **Assembly** : la composition. Les providers Dishka qui disent comment construire les services
  et les orchestrateurs. Elle construit, elle ne décide rien : aucune logique métier, aucun
  réglage de l'API. Elle ne construit pas de conteneur : chaque point d'entrée qui utilise Dishka
  construit le sien à partir de ces providers (`init_container()` dans `api/v1/api.py`).
- **Points d'entrée** (`api/`, `cli/`) : ils reçoivent une requête ou une commande, résolvent
  l'orchestrateur, le service ou, pour une lecture, le repository par leur conteneur, et
  délèguent. Ici, seule l'API existe ; un projet qui a une ligne de commande l'ajoute à côté, dans
  `cli/`, sur les mêmes providers.
  - **L'API suit la structure d'ecolopharm-sales :** `api.py` (l'app racine : elle monte la
    version et répond à `/health`), `settings.py` (`Settings` et `get_settings()`, les réglages de l'API seulement ; chaque usage
    appelle `get_settings()`, jamais une variable `settings` du module),
    `v1/api.py` (`init_container()`, `init_app(container)`, puis `container` et `app` au niveau du
    module), `v1/lifespan.py`, `v1/error_handling.py` et un fichier par routeur dans
    `v1/routers/`, chacun avec son `router = APIRouter(...)` et ses routes décorées.
  - **Une route reçoit son service par Dishka** (`DishkaRoute`, `FromDishka[...]`) et lui
    délègue. L'API n'utilise de Dishka que son intégration FastAPI et la construction de son
    conteneur ; aucun `Provider` n'y est déclaré, ils vivent dans `assembly/`.
  - **Une route qui ne fait que lire reçoit directement le repository,** comme un orchestrateur
    dont l'étape n'est qu'une lecture : pas de service qui ne ferait que relayer.
  - **L'API du gabarit n'authentifie personne.** Un projet ajoute sa vérification (une
    dépendance FastAPI sur le routeur) avant d'exposer une route.

## Contenu

Les classes à connaître, couche par couche.

### framework/

| Module | Ce qu'il fait | Classes clés |
|---|---|---|
| `multidigraph/` | Construire un graphe typé à partir d'enregistrements. | `MultiDiGraph`, `GraphBuilder` |
| `repository/` | Les contrats de lecture, d'écriture et des deux réunis, et leur substitut en mémoire pour les essais. | `AbstractReadRepository`, `AbstractWriteRepository`, `AbstractRepository`, `InMemoryRepository` |
| `repository/http/` | Une API HTTP distante ; un dialecte par fournisseur. | `ReadGateway`, `WriteGateway`, `Gateway`, `ZohoCoqlDialect`, `FidelioViewDialect` |
| `repository/nosql/mongo/` | MongoDB. | `MongoReadRepository`, `MongoWriteRepository`, `MongoRepository` |
| `repository/sql/` | Une base SQL, quel que soit le dialecte SQLAlchemy, sur des tables SQLModel et la session de l'appelant ; chaque écriture dans son point de sauvegarde, un refus de la base levé en `RecordRejectedError`, lu par le traducteur du dialecte (PostgreSQL, SQL Server, générique ailleurs). Extras : `sql`, plus le pilote, `postgres` (asyncpg) ou `sqlserver` (aioodbc). | `SqlReadRepository`, `SqlWriteRepository`, `SqlRepository`, `AbstractRejectionTranslator` |
| `repository/vectorstore/` | Recherche vectorielle et hybride. | `QdrantReadRepository`, `WeaviateReadRepository`, leurs pendants d'écriture, et `QdrantRepository`, `WeaviateRepository`, qui font les deux |

### services/

| Service | Ce qu'il montre |
|---|---|
| `record_matching/record_matching_service.py` | Comment un service se construit : la configuration au constructeur, les données dans la méthode. `RecordMatchingService.match(request)` reçoit un `MatchRequest` et dit à quelle `Organization` connue correspond le nom écrit dans un document, en comparant les noms normalisés. |
| `record_matching/crm_adapter.py` | Comment un adapter se construit : `CrmOrganization` lit un compte de CRM tel que son API le répond (identifiant numérique, nom imbriqué, adresse en parties) et rend une `Organization`, que le service reçoit sans connaître le CRM. |
| `batch_replacement_service.py` | `BatchReplacementService` : réécrire un lot à la place des enregistrements que tient le repository, en reprenant leurs ids. Un repository restreint (`SqlRepository(..., scope=...)`) limite le lot à sa portée. Ses essais le branchent sur Zoho, par `Gateway`. |

### orchestrators/

| Orchestrateur | Ce qu'il montre |
|---|---|
| `record_matching_orchestrator.py` | Comment un orchestrateur se construit : `RecordMatchingOrchestrator.match(name)` enchaîne les étapes d'un processus sans règle propre. Il lit les organisations connues par un repository, puis laisse `RecordMatchingService` décider. Le repository y est injecté directement, puisque l'étape n'est qu'une lecture. |

### models/

| Fichier | Ce qu'il montre |
|---|---|
| `organization.py` | `Organization` : l'entité du domaine que lisent l'adapter, le service et l'API. |
| `record_matching.py` | `MatchRequest`, ce que le service reçoit, et `MatchDecision`, ce qu'il rend ; l'API expose les deux. |

### assembly/ et api/

| Fichier | Ce qu'il montre |
|---|---|
| `assembly/service_provider.py` | `ServiceProvider` : le provider Dishka où un projet déclare ses services ; il construit ici `RecordMatchingService`. |
| `api/settings.py` | `Settings` et `get_settings()` : les réglages de l'API, lus de l'environnement et du `.env`. |
| `api/v1/api.py` | `init_container()`, le conteneur sur les providers de l'assembly ; `init_app(container)`, les routes et les erreurs en français. |
| `api/v1/routers/record_matching.py` | Une route qui reçoit son service par Dishka et lui délègue. |

## Ce qui entre ici

Un module entre s'il remplit **toutes** ces conditions.

1. **Générique.** Il ne nomme ni un client, ni un problème, ni une entité métier. Les
   docs et les docstrings s'expliquent avec des exemples que tout le monde comprend : un atelier,
   des tâches, des machines, des ressources. Jamais un cours, une pharmacie ou une facture.
2. **Éprouvé.** Il vient d'un vrai projet, où il est utilisé et testé. Quand plusieurs copies
   existent, on prend la plus récente : c'est d'ordinaire la plus évoluée.
3. **Porteur de savoir.** Il contient ce qu'un dev ne réécrirait pas facilement à partir d'une
   abstraction : un moteur, un protocole, une sémantique (filtres, pagination, liaison d'un
   modèle). Un adaptateur mince autour d'une API ou d'un format n'entre pas : il se réécrit sur
   l'abstraction.
4. **Autonome.** Il ne dépend d'aucun autre module de `framework/`. Ce dont il a besoin vit dans son
   dossier : une classe vit là où elle sert, comme `client/` dans `http/`.
   Entre dépôts, un module spécialisé peut lire un module d'ici (`framework/ml/graph/` de
   baseline-ml lit `framework/multidigraph/`) ; un module d'ici n'en lit jamais un d'un dépôt
   spécialisé.

### Ce qui n'entre pas

- **Un adaptateur mince autour d'une API tierce,** utilisé par un seul projet : le gateway montre
  déjà comment envelopper une API HTTP.
- **Un repository qui se réécrit en quelques lignes** sur un repository existant, comme lire et
  écrire un fichier entier sur `InMemoryRepository`.
- **Une ABC pour une seule implémentation.**
- **Le contenu métier d'un client** : ses services, adapters, orchestrateurs, prompts et `models`. Ce
  n'est pas une connaissance réutilisable.

## Règles d'architecture

`tests/architecture_test.py` (ArchUnitPython) les vérifie. Une règle qui échoue se corrige dans le
code, jamais en affaiblissant l'essai. Ses motifs partent du dossier du paquet, sous `src/` : le
nom d'un dossier parent ne change rien à ce qu'ils voient. Le nom du paquet n'y figure qu'une fois,
dans la fixture `package` : un dépôt spécialisé qui copie l'essai change ce nom et la liste des
dépôts qu'il n'importe pas.

- Tout fichier vit dans le dossier d'une couche. À la racine du paquet ne restent que son
  `__init__.py` et des fichiers qui ne sont pas du Python (`py.typed`) ; tout autre dossier, et
  tout `.py` qu'il contient, fait échouer l'essai.
- Les dépendances ne descendent que dans un sens :
  `api, cli → assembly → orchestrators → services → models → framework`.
  - Les points d'entrée (`api`, `cli`) sont au même rang : aucun n'importe l'autre.
  - Un projet ajoute les siens à ce rang, dans `ORDER` (un `worker` à côté de `api` et de `cli`).
- Les modules de `framework/` ne s'importent pas entre eux, `from <paquet>.framework import
  <module>` compris.
- Aucun import d'un dépôt spécialisé (`baseline_optimization`, `baseline_ml`, `baseline_llm`).
- Aucun cycle d'import.
- Aucun import d'un paquet client (`ecolopharm_sales`, `cegep_mv_schedule`, etc.).
- Un fichier `abstract_<nom>.py` définit la classe `Abstract…<Nom>`.
- **Un repository lit, écrit, ou les deux.** Trois contrats : `AbstractReadRepository` lit,
  `AbstractWriteRepository` écrit, `AbstractRepository` hérite des deux.
  - Chaque support de repository fournit trois classes : une de lecture, une d'écriture, et une
    qui fait les deux en héritant des deux premières. `MongoRepository` hérite de
    `MongoWriteRepository` et de `MongoReadRepository`, et de même pour `SqlRepository`,
    `QdrantRepository`, `WeaviateRepository` et `Gateway` (HTTP). `InMemoryRepository` implémente directement
    `AbstractRepository`.
  - Un appelant qui lit et écrit construit un seul objet : `MongoRepository(Machine, collection)`.
    Les classes de lecture et d'écriture restent pour celui qui ne fait que l'un des deux.
  - Une capacité propre au support va du côté où elle sert : `hybrid_search` est une lecture.
  - L'administration du support a sa propre classe (`WeaviateCollection`, `QdrantCollection`).

## Conventions de nommage

- Une ABC qui a de vraies implémentations dans d'autres fichiers vit dans `abstract_<nom>.py` et
  s'appelle `Abstract<Nom>` : `AbstractReadRepository`, `AbstractQueryDialect`. Une famille définie
  dans un seul fichier garde son nom nu (`Filter` dans `query_filters.py`).
- Un fichier porte le nom de sa classe principale : `in_memory_repository.py` →
  `InMemoryRepository`.
- Un dossier qui ne contiendrait qu'un fichier est remplacé par ce fichier.
- Un système externe a son dossier, nommé par le fournisseur ; ses classes portent ce nom :
  `http/zoho/` → `ZohoCoqlDialect`.
- Un `__init__.py` ne contient que sa docstring : pas d'import, pas de `__all__`. On importe une
  classe depuis le fichier qui la définit. Une réexportation chargerait tous les fichiers du paquet,
  avec leurs dépendances : importer le lecteur Qdrant chargeait `httpx` par le `__init__` de
  `repository/`. Sans `__all__`, un import dans `__init__` n'exporte d'ailleurs rien sous mypy strict.
- Les essais suivent l'arborescence du code.

## Ajouter un module

1. **Copier d'abord, sans réécrire,** depuis le projet d'origine. Noter l'origine (dépôt, branche,
   commit, chemin) dans `docs/SOURCES.txt`.
2. **Ranger** le code dans `framework/<sujet>/` et ses essais dans `tests/framework/<sujet>/`, en suivant les
   conventions de nommage.
3. **Relire contre `docs/rules.md`** classe par classe. Ce qui ne passe pas est réécrit
   ou retiré, jamais seulement déplacé.
4. **Retirer le couplage au projet d'origine** : noms de client, exemples métier, imports du
   `models` du client (un enregistrement générique partagé va dans `models/`).
5. **Faire revoir** par un agent indépendant (`@dev`). Les essais, y compris ceux
   d'architecture, doivent passer.

Un exemple de service, d'adapter ou d'orchestrateur montre **comment** on construit : la configuration et les
dépendances au constructeur, les données dans les méthodes. Il ne contient aucune règle métier
d'un client.

## État

- **Provenance :** chaque fichier est copié d'un projet ; `docs/SOURCES.txt` en donne l'origine.
- **Vérifier :** `uv sync --all-extras`, puis `uv run poe lint` (tous les hooks pre-commit : ruff,
  mypy, codespell, detect-secrets…) et `uv run poe test` (pytest sous coverage, au moins 95 %). La CI
  (`.github/workflows/test.yml`) lance les deux, en Python 3.12 et 3.14. Une fois par clone :
  `uv run pre-commit install`.
- **SQL Server :** les essais de `repository/sql/` tournent sur PostgreSQL et sur SQL Server. Sans
  le pilote « ODBC Driver 18 for SQL Server », ceux de SQL Server sont sautés ; la CI l'a et les
  exige. En local (Linux), `uv run poe test-sqlserver` les lance dans l'image de
  `docker/sqlserver-tests.Dockerfile`, qui a le pilote :

  ```bash
  docker build --tag baseline-common-sqlserver-tests --file docker/sqlserver-tests.Dockerfile docker
  docker run --rm --network host --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume "$PWD":/repo --volume baseline-common-uv-cache:/root/.cache/uv \
    --env TESTCONTAINERS_HOST_OVERRIDE=localhost baseline-common-sqlserver-tests \
    uv run --locked --all-extras --quiet pytest -p no:cacheprovider tests/framework/repository/sql
  ```
- **Une application sur SQL Server :** une colonne texte unique ou indexée a une longueur
  (`max_length`) ; `ondelete="RESTRICT"` n'existe pas (le défaut, `NO ACTION`, refuse de même) ;
  une contrainte ou un index unique n'admet qu'un seul `NULL` ; un texte hors Latin-1 demande
  `NVARCHAR` (`sa_type=Unicode`). Les contraintes portent un nom (`naming_convention`) : le
  traducteur en tire les colonnes, que SQL Server ne nomme pas. Les valeurs d'une référence
  brisée restent inconnues sur SQL Server.
- **Pipeline documentaire :** les prompts rendus neutres n'ont pas encore tourné contre un vrai
  modèle ; ils n'ont pas démontré qu'ils reproduisent la qualité mesurée dans le projet d'origine.
