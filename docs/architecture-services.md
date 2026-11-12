# Mode architectural — Data Models + Services

Les data models sont uniquement de la donnée. La logique métier vit dans les services. Les flots complexes sont coordonnés par des orchestrateurs.

## Principes SOLID

| Principe | Application |
|----------|-------------|
| **S — Single Responsibility** | Une classe = une responsabilité. Si la description contient « et », c'est deux classes. Un service cible une seule entité de données. |
| **O — Open/Closed** | Ouvert à l'extension (nouvelles implémentations d'interface), fermé à la modification (le code existant ne change pas). |
| **L — Liskov Substitution** | Toute implémentation d'une interface est interchangeable avec une autre. Pas de surprises. |
| **I — Interface Segregation** | Interfaces minimales et ciblées par consommateur. Si un consommateur n'utilise que 2 méthodes sur 10, l'interface est trop large — découper. |
| **D — Dependency Inversion** | Dépendre des abstractions (des ABC, jamais un `Protocol` : voir `docs/rules.md`), jamais des implémentations concrètes. Injection par constructeur. |

## Types de classes

- **Data Models** — objets de données purs, sans logique métier. Représentent les entités du domaine sous forme de structures sérialisables. Ceux que plusieurs services ou couches partagent vivent dans `data_schema/` ; celui qu'une seule classe utilise reste dans son fichier (définition complète dans le `README.md`).
- **Services** — portent la logique métier. Chaque service cible **une seule entité de données** (ex : `UserService` gère les `User`).
- **Orchestrateurs** — coordonnent plusieurs services quand un flot implique plusieurs entités. Un orchestrateur par flot métier (ex : `OnboardingOrchestrator` appelle `UserService`, `TeamService`, `NotificationService`).
- **Repositories** — sérialisent et persistent les data models. Un repository par entité.
- **Adapters** — isolent chaque système externe derrière une interface

## Noyau d'un module

Le noyau contient les **data models** (données pures), les **services** (logique métier) et les **orchestrateurs** (coordination de flots). Les data models n'ont pas de comportement — ils sont manipulés par les services.

```
┌──────────────────────────────────────┐
│  Implémentations                     │  ← Controllers API, SqlRepository, HttpClient
├──────────────────────────────────────┤
│  Interfaces typées (contrats)        │  ← Repository, Client, Notifier…
├──────────────────────────────────────┤
│  Noyau                               │  ← Data models, services, orchestrateurs
└──────────────────────────────────────┘
```

- **Interfaces typées** — chaque dépendance externe du noyau est abstraite par une interface au nom spécifique : `UserRepository`, `EmailNotifier`, `PaymentClient`, etc. Pas de classe générique « Port ».
- **Implémentations** — classes concrètes injectées par DI (contrôleurs API, repositories BD, clients HTTP).

## Diagramme de classes (template Mermaid)

```mermaid
classDiagram
    class {DataModel} {
        +{attribut}: {type}
    }
    class {ServiceInterface} {
        <<interface>>
        +{methode}({params}) {returnType}
    }
    class {ServiceName} {
        +{methode}({params}) {returnType}
    }
    class {OrchestratorName} {
        +{methodeDeFlot}({params}) {returnType}
    }
    class {RepositoryInterface} {
        <<interface>>
        +save({model}: {DataModel}) void
        +find_by_id({id}: {Type}) {DataModel}
    }
    class {RepositoryImpl} {
        +save({model}: {DataModel}) void
        +find_by_id({id}: {Type}) {DataModel}
    }
    {ServiceName} ..|> {ServiceInterface} : implémente
    {ServiceName} --> {DataModel} : manipule
    {ServiceName} --> {RepositoryInterface} : dépend de
    {OrchestratorName} --> {ServiceInterface} : coordonne
    {RepositoryImpl} ..|> {RepositoryInterface} : implémente
```

## Template — Section Classes (dans la spec de module)

### {ClassName}

**Responsabilité** : {Une seule phrase — si elle contient « et », c'est deux classes}
**Type** : {Data Model / Service / Orchestrateur / Repository / Adapter / Interface}

**Méthodes publiques** : (omettre pour les Data Models — ils n'exposent que des attributs)

```
{methode}({param}: {Type}) → {ReturnType}
```

- **Pré-conditions** : {ce qui doit être vrai avant l'appel}
- **Post-conditions** : {ce qui est garanti après l'appel}
- **Erreurs possibles** : {exceptions et quand elles surviennent}

## Patterns courants

Ne lister que les patterns **effectivement utilisés** avec leur justification. Ne pas forcer un pattern qui n'apporte pas de valeur.

- **Repository** — abstraction de la persistance, un par entité
- **Service Layer** — logique métier par entité, point d'entrée pour les cas d'utilisation
- **Orchestrator** — coordination de plusieurs services pour un flot multi-entités
- **Adapter/Client** — isolation d'un système externe derrière une interface interne
- **Dependency Injection** — câblage des services, repositories et adapters dans les endpoints
- **Strategy** — plusieurs algorithmes interchangeables pour un même comportement
- **Observer/Events** — découplage entre producteur et consommateur d'événements

## Anti-patterns

- **God service** — un service qui gère plusieurs entités → découper en un service par entité
- **God orchestrateur** — un orchestrateur qui gère tous les flots → un orchestrateur par flot
- **Logique dans les data models** — les data models sont de la donnée pure, la logique va dans les services
- **Logique dans les endpoints** — les endpoints injectent et délèguent, ils ne contiennent pas de logique métier
- **Dépendances circulaires** — A → B → A → extraire la partie commune
- **Service sans entité** — un service doit cibler une entité claire, sinon c'est probablement un orchestrateur
