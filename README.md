# baseline-core

## Raison d'être

baseline-core est la base de connaissances commune des développeurs de Baseline. On y trouve le
code qui a fait ses preuves dans un projet et qui servira au suivant :
- des briques génériques (`core/`) ;
- les enregistrements que les couches s'échangent (`data_schema/`) ;
- la façon d'enchaîner des services pour un flot (`orchestrator/`) ;
- la façon de construire un service par-dessus `core/` (`service/`).

Un projet généré par
[baseline-app-template](https://github.com/Baseline-quebec/baseline-app-template) choisit les
modules dont il a besoin, les copie, et en possède ensuite le code.

## Documentation

| Document | Contenu |
|---|---|
| [`docs/rules.md`](docs/rules.md) | **Les 29 règles de revue du code.** Tout code qui entre ici les respecte ; ce README les cite par numéro (« règle 5 »). |
| [`docs/architecture-services.md`](docs/architecture-services.md) | Les couches et comment un service, un orchestrateur et un repository se construisent. |
| [`docs/optimization-engine.md`](docs/optimization-engine.md) | Le moteur d'optimisation : écrire un problème, le résoudre. |
| [`docs/document-pipeline.md`](docs/document-pipeline.md) | La démarche documentaire : d'une source aux entités qu'un schéma demande. |
| [`docs/SOURCES.txt`](docs/SOURCES.txt) | La provenance de chaque fichier : dépôt, branche, commit, chemin. |

## Arborescence

```
docs/                            la documentation ci-dessus
iac/                             infrastructure as code (vide pour l'instant)
src/baseline_core/
├── api/v1/                      squelette FastAPI
├── core/                        briques génériques
│   ├── llm/
│   │   ├── document_extraction/ extraire d'un document ce qu'un schéma demande
│   │   └── evalkit/             évaluer la sortie d'un LLM
│   ├── ml/                      données, modèles, entraînement, inférence
│   ├── multidigraph/            graphe orienté multiple typé
│   ├── optimization/            écrire un problème d'optimisation, le résoudre
│   └── repository/              lire ou écrire des enregistrements
│       ├── gateway/             une API HTTP distante (Zoho, Fidelio)
│       ├── nosql/mongo/         MongoDB
│       ├── vectorstore/         Qdrant, Weaviate
│       ├── abstract_repository.py
│       └── in_memory_repository.py
├── data_schema/                 enregistrements partagés entre couches (vide)
├── orchestrator/                un fichier par flot
│   ├── document_extraction_orchestrator.py
│   └── planning_orchestrator.py
└── service/                     un service par entité ou capacité
    ├── ml/                      détecter des boîtes dans une image
    ├── optimization/            résoudre un problème d'optimisation
    ├── record_matching/agent/   comment un agent se construit
    └── batch_replacement_service.py
tests/                           même arborescence que src/ ; architecture_test.py
```

## Définitions

Dans l'ordre de l'arborescence.

- **Module core** : du code générique, sous `core/`. Il ne nomme ni un client ni un problème
  (règle 9).
- **Repository** : lit **ou** écrit un seul type d'enregistrement, sans règle métier. Il est
  injecté dans un service, ou dans un orchestrateur quand l'étape n'est qu'une lecture ou une
  écriture (pas de service qui ne ferait que relayer, règle 7).
- **data_schema** : les enregistrements Pydantic qu'une application partage entre ses couches
  (`api`, `orchestrator`, `service`) ou entre plusieurs services. C'est le vocabulaire commun de
  l'application, la couche juste au-dessus de `core/`.
  - **Ce qui y entre :** un enregistrement lu ou produit par plus d'un service ou plus d'une
    couche (une entité du domaine, une requête ou une réponse que l'API expose et qu'un service
    produit). Un enregistrement qu'une seule classe utilise reste dans le fichier de cette classe
    (règle 8) : la sortie d'un agent dans son `output.py`, la configuration d'un service à côté de
    lui.
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
  et les dépendances, ses méthodes reçoivent les données (règle 1). Un service a son dossier
  `service/<sujet>/` dès qu'un second fichier n'appartient qu'à lui (agent, solveur, config).

## Contenu

Les classes à connaître, couche par couche.

### core/

| Module | Ce qu'il fait | Classes clés |
|---|---|---|
| `llm/document_extraction/` | Lire un PDF ou un classeur, en trouver les sections, garder celles qui comptent, en extraire les entités. | `DocumentParser`, `IndexingService`, `ClassificationService`, `ExtractionService` |
| `llm/evalkit/` | Mesurer la qualité d'une sortie de LLM sur des cas, suivie dans MLflow. | `AbstractMetric`, `Evaluator`, `EvalRun` |
| `ml/` | Entraîner un modèle (PyTorch Lightning), le suivre (MLflow), l'exporter et l'exécuter. | `TrainingRun`, `AbstractModelRunner`, `OnnxExporter` |
| `ml/graph/` | Lire un graphe comme entrée de modèle (PyTorch Geometric). Dépend de `multidigraph`. | `HeteroDataConverter` |
| `multidigraph/` | Construire un graphe typé à partir d'enregistrements. | `MultiDiGraph`, `GraphBuilder` |
| `optimization/` | Écrire un problème une fois, le lier à des données, le pré-vérifier, le résoudre. | `AbstractModelDefinition`, `ModelCompiler`, `InfeasibilityCheck`, `KernelSolver`, `OutputMapper` |
| `repository/` | Les contrats de lecture et d'écriture, et leur substitut en mémoire pour les essais. | `AbstractReadRepository`, `AbstractWriteRepository`, `InMemoryRepository` |
| `repository/gateway/` | Une API HTTP distante ; un dialecte par fournisseur. | `ReadGateway`, `WriteGateway`, `ZohoCoqlDialect`, `FidelioViewDialect` |
| `repository/nosql/mongo/` | MongoDB. | `MongoReadRepository`, `MongoWriteRepository` |
| `repository/vectorstore/` | Recherche vectorielle et hybride. | `QdrantReadRepository`, `WeaviateReadRepository` et leurs pendants d'écriture |

### orchestrator/

| Fichier | Flot |
|---|---|
| `document_extraction_orchestrator.py` | Un fichier → `DocumentParser`, `IndexingService`, `ClassificationService`, `ExtractionService`. |
| `planning_orchestrator.py` | Lire un problème, le résoudre avec `OptimizationService`, écrire le plan. |

### service/

| Service | Ce qu'il montre |
|---|---|
| `ml/` | `Detector` : un service sur un modèle exécuté par `core/ml/`. |
| `optimization/` | `OptimizationService` : lier, pré-vérifier, résoudre, remplir la sortie. |
| `record_matching/agent/` | Comment un agent se construit : `agent.py`, `deps.py`, `output.py`, `prompt.py`. Ici, à quel enregistrement connu correspond un nom écrit dans un document. |
| `batch_replacement_service.py` | `BatchReplacementService` : réécrire un lot à la place du précédent de même portée. Ses essais le branchent sur Zoho : l'exemple d'un service sur un gateway. |

## Ce qui entre ici

Un module entre s'il remplit **toutes** ces conditions.

1. **Générique.** Il ne nomme ni un client, ni un problème, ni une entité métier (règle 9). Les
   docs et les docstrings s'expliquent avec des exemples que tout le monde comprend : un atelier,
   des tâches, des machines, des ressources. Jamais un cours, une pharmacie ou une facture.
2. **Éprouvé.** Il vient d'un vrai projet, où il est utilisé et testé. Quand plusieurs copies
   existent, on prend la plus récente : c'est d'ordinaire la plus évoluée.
3. **Porteur de savoir.** Il contient ce qu'un dev ne réécrirait pas facilement à partir d'une
   abstraction : un moteur, un protocole, une sémantique (filtres, pagination, liaison d'un
   modèle). Un adaptateur mince autour d'une API ou d'un format n'entre pas : il se réécrit sur
   l'abstraction.
4. **Autonome.** Il ne dépend d'aucun autre module de `core/`. Ce dont il a besoin vit dans son
   dossier : une classe vit là où elle sert (règle 8), comme `http_client/` dans `gateway/`.
   **Seule exception : une dépendance déclarée,** quand une partie d'un module travaille sur la
   structure d'un autre (`ml/graph/` convertit un `MultiDiGraph`). Elle est :
   - **isolée** dans un sous-dossier ; le reste du module ne l'importe pas et reste copiable seul ;
   - **à sens unique** : `multidigraph` n'importe jamais `ml` ;
   - **déclarée** dans `TestLayers.DECLARED` (`tests/architecture_test.py`) et dans l'extra du
     sous-dossier (`ml-graph` inclut `graph`). Un projet qui prend `ml/graph/` reçoit aussi
     `multidigraph/`.
   - **importée en haut du fichier,** jamais dans une fonction ni dans un `try` : une dépendance
     cachée échoue tard et échappe à mypy et aux essais d'architecture.

### Ce qui n'entre pas

- **Un adaptateur mince autour d'une API tierce,** utilisé par un seul projet : le gateway montre
  déjà comment envelopper une API HTTP.
- **Un repository qui se réécrit en quelques lignes** sur un repository existant, comme lire et
  écrire un fichier entier sur `InMemoryRepository`.
- **Une ABC pour une seule implémentation** (règle 5).
- **Le contenu métier d'un client** : ses services, orchestrateurs, prompts et `data_schema`. Ce
  n'est pas une connaissance réutilisable.

## Règles d'architecture

`tests/architecture_test.py` (ArchUnitPython) les vérifie. Une règle qui échoue se corrige dans le
code, jamais en affaiblissant l'essai.

- Les dépendances ne descendent que dans un sens :
  `api → orchestrator → service → data_schema → core`.
- Les modules de `core/` ne s'importent pas entre eux, sauf une dépendance déclarée (voir
  « Autonome »), et le reste du module n'importe jamais le sous-dossier qui la porte.
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
3. **Relire contre `docs/rules.md`** classe par classe (règle 26). Ce qui ne passe pas est réécrit
   ou retiré, jamais seulement déplacé.
4. **Retirer le couplage au projet d'origine** : noms de client, exemples métier, imports du
   `data_schema` du client (un enregistrement générique partagé va dans `data_schema/`).
5. **Faire revoir** par un agent indépendant (`@dev`, règle 13). Les essais, y compris ceux
   d'architecture, doivent passer.

Un exemple de service ou d'orchestrateur montre **comment** on construit : la configuration et les
dépendances au constructeur, les données dans les méthodes. Il ne contient aucune règle métier
d'un client.

## État

- **Provenance :** chaque fichier est copié d'un projet ; `docs/SOURCES.txt` en donne l'origine.
- **Vérifier :** `uv sync --all-extras`, puis `uv run mypy`, `uv run ruff check src tests`,
  `uv run ruff format --check src tests` et `uv run pytest` passent tous.
- **Pipeline documentaire :** les prompts rendus neutres n'ont pas encore tourné contre un vrai
  modèle ; ils n'ont pas démontré qu'ils reproduisent la qualité mesurée dans le projet d'origine.
- **Trackers MLflow :** `llm/evalkit/tracking/` (évaluer une sortie de LLM) et `ml/tracking/`
  (suivre un entraînement) restent séparés : deux modules de `core/` ne s'importent pas, et ils ne
  font pas le même travail.
