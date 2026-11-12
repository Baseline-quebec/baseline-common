# baseline-common

## Raison d'être

baseline-common est la base de connaissances commune des développeurs de Baseline. On y trouve le
code qui a fait ses preuves dans un projet et qui servira au suivant :
- des briques génériques (`core/`) ;
- les enregistrements que les couches s'échangent (`data_schema/`) ;
- la façon d'enchaîner des services pour un flot (`orchestrator/`) ;
- la façon de construire un service par-dessus `core/` (`service/`).

Un projet généré par
[baseline-app-template](https://github.com/Baseline-quebec/baseline-app-template) choisit les
modules dont il a besoin, les copie, et en possède ensuite le code.

## Dépôts spécialisés

Trois modules ont quitté baseline-common pour leur propre dépôt. **Leur accès est réservé aux
administrateurs** de Baseline-quebec ; baseline-common reste ouvert à tous.

| Dépôt | Ce qu'il contient |
|---|---|
| [baseline-optimization](https://github.com/Baseline-quebec/baseline-optimization) | Le moteur d'optimisation : écrire un problème, le lier, le pré-vérifier, le résoudre ; ses règles propres. |
| [baseline-ml](https://github.com/Baseline-quebec/baseline-ml) | Entraîner, suivre, exporter et exécuter un modèle ; `graph/` lit `multidigraph/` d'ici. |
| [baseline-llm](https://github.com/Baseline-quebec/baseline-llm) | L'extraction de documents, l'évaluation des sorties, le coût des jetons, et comment un agent se construit. |

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
iac/                             infrastructure as code (vide pour l'instant)
src/baseline_common/
├── api/v1/                      squelette FastAPI
├── core/                        briques génériques
│   ├── multidigraph/            graphe orienté multiple typé
│   └── repository/              lire ou écrire des enregistrements
│       ├── gateway/             une API HTTP distante (Zoho, Fidelio)
│       ├── nosql/mongo/         MongoDB
│       ├── vectorstore/         Qdrant, Weaviate
│       ├── abstract_repository.py
│       └── in_memory_repository.py
├── data_schema/                 enregistrements partagés entre couches (vide)
├── orchestrator/                un fichier par flot (vide ici ; exemples dans les dépôts spécialisés)
└── service/                     un service par entité ou capacité
    ├── record_matching/matcher/ comment un service se construit
    └── batch_replacement_service.py
tests/                           même arborescence que src/ ; architecture_test.py
```

## Définitions

Dans l'ordre de l'arborescence.

- **Module core** : du code générique, sous `core/`. Il ne nomme ni un client ni un problème.
- **Repository** : lit **ou** écrit un seul type d'enregistrement, sans règle métier. Il est
  injecté dans un service, ou dans un orchestrateur quand l'étape n'est qu'une lecture ou une
  écriture (pas de service qui ne ferait que relayer).
- **data_schema** : les enregistrements Pydantic qu'une application partage entre ses couches
  (`api`, `orchestrator`, `service`) ou entre plusieurs services. C'est le vocabulaire commun de
  l'application, la couche juste au-dessus de `core/`.
  - **Ce qui y entre :** un enregistrement lu ou produit par plus d'un service ou plus d'une
    couche (une entité du domaine, une requête ou une réponse que l'API expose et qu'un service
    produit). Un enregistrement qu'une seule classe utilise reste dans le fichier de cette classe :
    la sortie d'un agent dans son `output.py`, la configuration d'un service à côté de lui.
  - **Ce qu'il contient :** des champs, des validateurs d'invariants (un identifiant bien formé,
    une intégrité référentielle) et des propriétés dérivées simples. Aucune entrée-sortie, aucun
    appel de service, aucune règle métier lourde : une transformation de plusieurs centaines de
    lignes est un service, même si elle ne produit qu'un enregistrement.
  - **Ce qu'il importe :** `core/` et Pydantic seulement. Il n'importe jamais `service/`,
    `orchestrator/` ni `api/`, et `core/` ne l'importe jamais (un module de `core/` reste
    copiable seul).
  - **Ici,** il est vide : les enregistrements génériques vivent dans leur module de `core/`. Dans
    un projet, il porte ceux du domaine du client.
- **Orchestrateur** : enchaîne plusieurs services pour un flot, sans règle métier propre. C'est un
  seul fichier, `orchestrator/<flot>_orchestrator.py`. C'est là que des modules de `core/`, qui
  s'ignorent entre eux, se rencontrent.
- **Service** : la logique d'une entité ou d'une capacité. Son constructeur reçoit la configuration
  et les dépendances, ses méthodes reçoivent les données. Un service a son dossier
  `service/<sujet>/` dès qu'un second fichier n'appartient qu'à lui (agent, solveur, config).

## Contenu

Les classes à connaître, couche par couche.

### core/

| Module | Ce qu'il fait | Classes clés |
|---|---|---|
| `multidigraph/` | Construire un graphe typé à partir d'enregistrements. | `MultiDiGraph`, `GraphBuilder` |
| `repository/` | Les contrats de lecture et d'écriture, et leur substitut en mémoire pour les essais. | `AbstractReadRepository`, `AbstractWriteRepository`, `InMemoryRepository` |
| `repository/gateway/` | Une API HTTP distante ; un dialecte par fournisseur. | `ReadGateway`, `WriteGateway`, `ZohoCoqlDialect`, `FidelioViewDialect` |
| `repository/nosql/mongo/` | MongoDB. | `MongoReadRepository`, `MongoWriteRepository` |
| `repository/vectorstore/` | Recherche vectorielle et hybride. | `QdrantReadRepository`, `WeaviateReadRepository` et leurs pendants d'écriture |

### service/

| Service | Ce qu'il montre |
|---|---|
| `record_matching/matcher/` | Comment un service se construit : `deps.py`, `output.py`, `matcher.py`. À quel enregistrement connu correspond un nom écrit dans un document, décidé par des règles. |
| `batch_replacement_service.py` | `BatchReplacementService` : réécrire un lot à la place du précédent de même portée. Ses essais le branchent sur Zoho : l'exemple d'un service sur un gateway. |

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
4. **Autonome.** Il ne dépend d'aucun autre module de `core/`. Ce dont il a besoin vit dans son
   dossier : une classe vit là où elle sert, comme `http_client/` dans `gateway/`.
   Entre dépôts, un module spécialisé peut lire un module d'ici (`graph/` de baseline-ml lit
   `multidigraph/`) ; un module d'ici n'en lit jamais un d'un dépôt spécialisé.

### Ce qui n'entre pas

- **Un adaptateur mince autour d'une API tierce,** utilisé par un seul projet : le gateway montre
  déjà comment envelopper une API HTTP.
- **Un repository qui se réécrit en quelques lignes** sur un repository existant, comme lire et
  écrire un fichier entier sur `InMemoryRepository`.
- **Une ABC pour une seule implémentation.**
- **Le contenu métier d'un client** : ses services, orchestrateurs, prompts et `data_schema`. Ce
  n'est pas une connaissance réutilisable.

## Règles d'architecture

`tests/architecture_test.py` (ArchUnitPython) les vérifie. Une règle qui échoue se corrige dans le
code, jamais en affaiblissant l'essai.

- Les dépendances ne descendent que dans un sens :
  `api → orchestrator → service → data_schema → core`.
- Les modules de `core/` ne s'importent pas entre eux.
- Aucun import d'un dépôt spécialisé (`baseline_optimization`, `baseline_ml`, `baseline_llm`).
- Aucun cycle d'import.
- Aucun import d'un paquet client (`ecolopharm_sales`, `cegep_mv_schedule`, etc.).
- Un fichier `abstract_<nom>.py` définit la classe `Abstract…<Nom>`.
- **La lecture et l'écriture ne se mélangent pas.** Une abstraction lit
  (`AbstractReadRepository`) ou écrit (`AbstractWriteRepository`), jamais les deux.
  - Chaque support fournit une classe de lecture et une classe d'écriture : `ReadGateway` et
    `WriteGateway`, `WeaviateReadRepository` et `WeaviateWriteRepository`, `QdrantReadRepository`
    et `QdrantWriteRepository`, `MongoReadRepository` et `MongoWriteRepository`.
  - Une capacité propre au support va du côté où elle sert : `hybrid_search` est une lecture.
  - L'administration du support a sa propre classe (`WeaviateCollection`, `QdrantCollection`).
  - Seul `InMemoryRepository`, le substitut des essais, implémente les deux contrats dans une
    classe, parce qu'il n'a qu'un état.

## Conventions de nommage

- Une ABC qui a de vraies implémentations dans d'autres fichiers vit dans `abstract_<nom>.py` et
  s'appelle `Abstract<Nom>` : `AbstractReadRepository`, `AbstractQueryDialect`. Une famille définie
  dans un seul fichier garde son nom nu (`Filter` dans `query_filters.py`).
- Un fichier porte le nom de sa classe principale : `in_memory_repository.py` →
  `InMemoryRepository`.
- Un dossier qui ne contiendrait qu'un fichier est remplacé par ce fichier.
- Un système externe a son dossier, nommé par le fournisseur ; ses classes portent ce nom :
  `gateway/zoho/` → `ZohoCoqlDialect`.
- Un `__init__.py` ne contient que sa docstring : pas d'import, pas de `__all__`. On importe une
  classe depuis le fichier qui la définit. Une réexportation chargerait tous les fichiers du paquet,
  avec leurs dépendances : importer le lecteur Qdrant chargeait `httpx` par le `__init__` de
  `repository/`. Sans `__all__`, un import dans `__init__` n'exporte d'ailleurs rien sous mypy strict.
- Les essais suivent l'arborescence du code.

## Ajouter un module

1. **Copier d'abord, sans réécrire,** depuis le projet d'origine. Noter l'origine (dépôt, branche,
   commit, chemin) dans `docs/SOURCES.txt`.
2. **Ranger** le code dans `core/<sujet>/` et ses essais dans `tests/core/<sujet>/`, en suivant les
   conventions de nommage.
3. **Relire contre `docs/rules.md`** classe par classe. Ce qui ne passe pas est réécrit
   ou retiré, jamais seulement déplacé.
4. **Retirer le couplage au projet d'origine** : noms de client, exemples métier, imports du
   `data_schema` du client (un enregistrement générique partagé va dans `data_schema/`).
5. **Faire revoir** par un agent indépendant (`@dev`). Les essais, y compris ceux
   d'architecture, doivent passer.

Un exemple de service ou d'orchestrateur montre **comment** on construit : la configuration et les
dépendances au constructeur, les données dans les méthodes. Il ne contient aucune règle métier
d'un client.

## État

- **Provenance :** chaque fichier est copié d'un projet ; `docs/SOURCES.txt` en donne l'origine.
- **Vérifier :** `uv sync --all-extras`, puis `uv run poe lint` (tous les hooks pre-commit : ruff,
  mypy, codespell, detect-secrets…) et `uv run poe test` (pytest sous coverage). La CI
  (`.github/workflows/test.yml`) lance les deux, en Python 3.12 et 3.14. Une fois par clone :
  `uv run pre-commit install`.
- **Pipeline documentaire :** les prompts rendus neutres n'ont pas encore tourné contre un vrai
  modèle ; ils n'ont pas démontré qu'ils reproduisent la qualité mesurée dans le projet d'origine.
