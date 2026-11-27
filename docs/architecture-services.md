# Mode architectural — Data Models + Services

Les data models portent la donnée et ses invariants. La logique métier vit dans les services. Les flots complexes sont coordonnés par des orchestrateurs.

## Principes SOLID

| Principe | Application |
|----------|-------------|
| **S — Single Responsibility** | Une classe = une responsabilité. Si la description contient « et », c'est deux classes. Un service porte une seule entité ou une seule capacité. |
| **O — Open/Closed** | Ouvert à l'extension (une nouvelle sous-classe d'un contrat), fermé à la modification (le code existant ne change pas). |
| **L — Liskov Substitution** | Toute sous-classe d'un contrat est interchangeable avec une autre. Pas de surprises. |
| **I — Interface Segregation** | Contrats minimaux et ciblés par consommateur. Si un consommateur n'utilise que 2 méthodes sur 10, le contrat est trop large — découper, comme `AbstractReadRepository` et `AbstractWriteRepository`. |
| **D — Dependency Inversion** | Injection par constructeur. On dépend du contrat quand il existe, sinon de la classe concrète : un contrat s'écrit en `ABC` et n'existe qu'avec plusieurs implémentations, ou pour un point d'extension. |

## Types de classes

- **Data Models** — les enregistrements Pydantic du domaine : des champs, des validateurs d'invariants et des propriétés dérivées simples, sans entrée-sortie ni règle métier lourde. Ceux que plusieurs services ou couches partagent vivent dans `models/` ; celui qu'une seule classe utilise reste dans son fichier (définition complète dans le `README.md`).
- **Services** — portent la logique métier d'**une entité ou d'une capacité** (ex : `UserService` gère les `User` ; `RecordMatchingService` rapproche un nom écrit d'une organisation connue). Le constructeur reçoit la configuration et les dépendances, les méthodes reçoivent les données.
- **Orchestrateurs** — le pipeline d'un processus : ils enchaînent ses étapes, des services, et un repository quand l'étape n'est qu'une lecture ou une écriture. Un orchestrateur par flot métier (ex : `OnboardingOrchestrator` appelle `UserService`, `TeamService`, `NotificationService` ; `RecordMatchingOrchestrator` lit les organisations par un repository, puis appelle `RecordMatchingService`). Une ou deux méthodes publiques ; une de plus demande une raison (une autre entrée exposée à l'API).
- **Repositories** — lisent ou écrivent les data models, sans règle métier. Un repository par entité, sur les contrats de `framework/repository/`.
- **Adapters** — lisent un enregistrement dans la forme d'un système externe et rendent celui du domaine. Ils vivent dans le dossier du service, `services/<sujet>/<source>_adapter.py`, et sont optionnels : dans 90 à 95 % des cas, un schéma de source à `AliasPath` suffit (définition complète dans le `README.md`).
- **Assembly** — la composition : les providers Dishka qui disent comment construire les services et les orchestrateurs. Chaque point d'entrée (`api/`, `cli/`) construit son conteneur à partir d'eux. Elle construit et ne décide rien (définition complète dans le `README.md`).

## Noyau d'un module

Le noyau contient les **data models** (la donnée et ses invariants), les **services** (logique métier) et les **orchestrateurs** (coordination de flots). Les data models ne portent aucune règle métier lourde — les services les manipulent.

```
┌──────────────────────────────────────┐
│  Implémentations                     │  ← SqlRepository, MongoRepository, Gateway
├──────────────────────────────────────┤
│  Contrats (ABC)                      │  ← AbstractReadRepository, AbstractWriteRepository…
├──────────────────────────────────────┤
│  Noyau                               │  ← Data models, services, orchestrateurs
└──────────────────────────────────────┘
```

- **Contrats** — une dépendance du noyau passe par une `ABC` au nom précis quand il a plusieurs implémentations, ou marque un point d'extension. Les repositories passent par ceux de `framework/repository/`. Sinon, le noyau reçoit la classe concrète. Pas de classe générique « Port ».
- **Implémentations** — classes concrètes que les providers de `assembly/` construisent et injectent (repositories, gateways HTTP, services).

## Diagramme de classes (template Mermaid)

Le modèle ne dessine aucun contrat de service : une `ABC` de service n'apparaît qu'avec plusieurs implémentations, ou pour un point d'extension.

```mermaid
classDiagram
    class {DataModel} {
        +{attribut}: {type}
    }
    class {ServiceName} {
        +{methode}({model}: {DataModel}) {returnType}
    }
    class {OrchestratorName} {
        +{methodeDeFlot}({params}) {returnType}
    }
    class AbstractRepository {
        <<abstract>>
        +insert_one({model}: {DataModel}) str
        +get_by_id({id}: str) {DataModel}
    }
    class {RepositoryImpl} {
        +insert_one({model}: {DataModel}) str
        +get_by_id({id}: str) {DataModel}
    }
    {ServiceName} --> {DataModel} : manipule
    {ServiceName} --> AbstractRepository : dépend de
    {OrchestratorName} --> {ServiceName} : coordonne
    {RepositoryImpl} --|> AbstractRepository : hérite
```

## Template — Section Classes (dans la spec de module)

### {ClassName}

**Responsabilité** : {Une seule phrase — si elle contient « et », c'est deux classes}
**Type** : {Data Model / Service / Orchestrateur / Repository / Adapter / Contrat (ABC)}

**Méthodes publiques** : (omettre pour les Data Models — ils n'exposent que des champs et des propriétés dérivées)

```
{methode}({param}: {Type}) → {ReturnType}
```

- **Pré-conditions** : {ce qui doit être vrai avant l'appel}
- **Post-conditions** : {ce qui est garanti après l'appel}
- **Erreurs possibles** : {exceptions et quand elles surviennent}

## Patterns courants

Ne lister que les patterns **effectivement utilisés** avec leur justification. Ne pas forcer un pattern qui n'apporte pas de valeur.

- **Repository** — lecture ou écriture d'un type d'enregistrement, sur les contrats de `framework/repository/`
- **Gateway** — une API HTTP distante derrière les mêmes contrats (`framework/repository/http/`)
- **Service Layer** — logique métier par entité ou capacité, point d'entrée pour les cas d'utilisation
- **Orchestrator** — coordination des étapes d'un processus : des services, un repository pour une simple lecture ou écriture
- **Adapter** — lecture d'un enregistrement dans la forme d'un système externe, rendu en enregistrement du domaine
- **Dependency Injection** — câblage des services et des repositories dans les endpoints, déclaré dans `assembly/`
- **Strategy** — plusieurs algorithmes interchangeables pour un même comportement (un dialecte de requête, un traducteur de refus)

## Anti-patterns

- **God service** — un service qui gère plusieurs entités ou capacités → un service par entité ou capacité
- **God orchestrateur** — un orchestrateur qui gère tous les flots → un orchestrateur par flot
- **Règle métier lourde dans les data models** — un data model valide ses invariants et dérive des propriétés simples ; une règle métier lourde (une transformation de plusieurs centaines de lignes), une entrée-sortie ou un appel de service va dans un service
- **Logique dans les endpoints** — les endpoints injectent et délèguent, ils ne contiennent pas de logique métier
- **Dépendances circulaires** — A → B → A → extraire la partie commune
- **Service relais** — un service qui ne fait que transmettre une lecture au repository : la route ou l'orchestrateur reçoit le repository directement
