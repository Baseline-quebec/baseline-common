# Moteur d'optimisation : historique

Ce fichier garde, mot pour mot, ce qui a quitté `docs/optimization-engine.md` quand ce document est
devenu une référence (2026-11-09). C'est l'historique du projet où le moteur est né (horaires de
cours d'un cégep) : décisions, mesures datées, revues, journal des demandes. Les chemins
(`apps/backend/src`, `docs/fixes/`, `modeling/`), les fichiers maîtres et les scripts cités
n'existent pas dans ce dépôt. Pour se servir du moteur, lire `docs/optimization-engine.md`.

Les deux extraits suivent ; chacun indique les lignes qu'il occupait dans l'ancien document. Les
sections 0 à 4 de l'extrait 1 ont été réécrites dans la référence sur un atelier ; elles restent
ici dans leur version d'origine, sur les cours.

---

<!-- Extrait 1 : ancien docs/optimization-engine.md, lignes 1 à 732 (sections 0 à 6 bis) -->

# Moteur de modélisation générique

Un modèle s'écrit une fois, sans données, avec des déclarations (ensembles, paramètres, familles,
décisions) et des règles. Il est compilé en un noyau Cython, celui que la recherche utilise, puis lié à
l'objet Pydantic du problème. Le moteur ne connaît aucun problème : les horaires de cours, un problème
de livraisons (`tests/solver/binding/model_compiler_test.py`, `TestDeliveries`) et un atelier
(`tests/solver/infeasibility/precheck_test.py`) ne servent qu'à l'essayer.

## 0. Où vit le code

Le moteur est le paquet `optimization/` ; il n'est **pas** importé dans `apps/backend/src`, et il
n'importe rien du projet (`tests/architecture_test.py` le vérifie). Ses dossiers reprennent les noms de
`apps/backend/src` (`problem_formulation` vient de `core/scheduling`, `solver` de `core/optimization`).

```
docs/fixes/
  optimization/
    problem_formulation/          comment un problème s'écrit
      model.py                    Model : déclarations, décisions, règles
      model_definition.py         ModelDefinition (ABC) : un modèle écrit contre un type de schéma
      schema_path.py              RecordedPath : le chemin qu'une lambda lit, vérifié à l'écriture
      expression.py               Expr et ses nœuds ; les opérateurs, en Literal, comme le code généré
      collection.py               Collection, FilteredSet, SortedList
      declaration.py              IndexSet, Parameter, SubsetFamily, Subset, Assignment, …
      term.py                     Term : une règle ou un terme d'objectif, nommé
    solver/                       comment un modèle écrit est résolu
      binding/                    un modèle rencontre ses données : compilé une fois, lié à chaque problème
        model_compiler.py         ModelCompiler : compile une fois, lie à chaque jeu de données ; BoundModel
        binder.py                 Binder : déplie chaque règle sur les données
        bound_collection.py       FixedSubset, PartitionPart, ComputedSubset : les collections liées
        schema_reader.py          SchemaReader : lit les valeurs dans l'objet Pydantic
      solution.py                 VariableAssignment, ModelSolution : ce qu'une résolution prend et rend, en clés
      output_mapper.py            OutputMapper : remplit la sortie qu'un problème déclare, sur une solution
      score.py                    Score : dur puis mou
      stop.py                     MaxIterations, MaxTime, NoImprovement, AnyOf : quand une recherche s'arrête
      kernel/
        kernel.py                 Kernel : le noyau écrit à la main (Cython « mode Python pur »)
        codegen.py                KernelSource : écrit la sous-classe générée du modèle
        model_kernel.py           ModelKernel : le gabarit de la sous-classe, que codegen remplit
        shape.py                  TermShape, Template : les formes que le code généré partage
        builder.py                KernelBuilder : compile (ou importe en Python pur), cache disque
      infeasibility/
        precheck.py               InfeasibilityCheck, PrecheckLevel : trois niveaux de preuve, délai réglable
        precheck_run.py           PrecheckRun : une exécution, ce que les règles dures lient et le domaine de chaque élément
        placement_search.py       ExhaustiveSearch, HeuristicSearch : la recherche d'un groupe
        fork_pool.py              ForkPool : les groupes indécis cherchés dans des copies du processus
        precheck_issue.py                PrecheckIssue : un constat
      search/                     la recherche du backend (détruire-réparer) portée sur le noyau
        index.py                  SearchIndex : domaines seuls, et les liens (Tie) que le modèle déclare
        solver.py                 KernelSolver, KernelSolverConfig : la boucle, gardant les coups pas pires
        state.py                  KernelState : le noyau, et une copie de son placement
        evaluator.py              KernelEvaluator : un coup compté autour des éléments déplacés
        destroy_repair.py         DestroyRepair : libère, replace au mieux, aligne les paires
        selector.py               DestroySelector, AlnsConfig : les 9 bras du backend, pondérés par roulette
    mining/                       après une extraction : quelles règles encoder, comment lire l'objectif
      plan.py                     Plan (lu dans le noyau ou chargé) ; PlanView : les clés des tables du
                                  planificateur, par rôle, et le découpage des jours
      findings.py                 Findings : le findings.json de l'extraction, typé, lu en candidats
      candidate.py                Candidate : une règle trouvée, la règle dure qu'elle pose (nommée d'après
                                  elle, sur les seuls éléments de sa clé) ; ConstrainedDefinition : un
                                  modèle et ses candidats
      sampler.py                  FeasibleWalk (marche sur le dur depuis le placement du noyau, dans les
                                  domaines de SearchIndex), KernelRestarts (KernelSolver, une graine chacun)
      constrain_loop.py           ConstrainLoop : encoder d'abord ce que le modèle explique le moins bien ;
                                  les plans sont jugés par le noyau, sur un modèle portant tous les candidats
      tradeoff.py                 ObjectiveTradeoff : deux objectifs, dans quel sens les plans ont bougé
  scripts/                        vérifications et mesures (python -m scripts.<nom>)
  tests/                          même découpage que optimization/
    optimization_service.py       OptimizationService, OptimizationResult : lier une fois, pré-vérifier, résoudre, remplir la sortie
    course_schedule.py            CourseSolution : le schéma de sortie des cours, sa seule configuration
    course_model.py               les règles des cours : un modèle d'essai, pas un livrable
    models.py                     petits modèles d'essai
    job_shop_model.py             un atelier : le même moteur sur un autre problème
    large_course_problem.py       LargeCourseProblem : un problème de cours à l'échelle (scale_precheck)
    parity/                       les comparaisons au backend : faisabilité, sortie, recherche,
                                  définition des règles (les scripts compare_* et prove_* les lancent)
    mining/                       ses essais ; l'atelier (shop.py), le banc à règles cachées
                                  (planted_bench.py) et le banc sur le fichier maître (real_bench.py)
```

```bash
cd docs/fixes
uv run --project ../../apps/backend pytest                                   # 170 tests
uv run --project ../../apps/backend ruff check --no-fix optimization tests
uv run --project ../../apps/backend mypy --config-file ../../apps/backend/pyproject.toml optimization tests
uv run --project ../../apps/backend python -m scripts.course_flow            # un script
```

Le banc à règles cachées vit dans `tests/mining/planted_bench.py` (ses règles cachées sont des
candidats posés par `ConstrainedDefinition`) ; ses scripts : `export_bench_tables` (`--with pandas`),
`constrain_loop_check`, `objective_tradeoff_check`. Le banc sur le fichier maître :
`tests/mining/real_bench.py`, lancé par `scripts/real_bench.py`.
Résultat négatif gardé : noter un candidat par
« les plans du modèle ressemblent davantage à ceux du planificateur une fois le candidat encodé » ne
distingue pas les vraies règles des fausses (encoder les 34 candidats ou les 10 vraies donne le même
écart).

> `ruff check` sans `--no-fix` **corrige** les fichiers (le `pyproject.toml` du backend a
> `fix = true`).

## 1. Écrire un modèle, le résoudre

```python
class SharedRoomDefinition(ModelDefinition[CourseProblemInstance]):  # une ABC : schema + define
    schema = CourseProblemInstance                         # le type des données, dans la classe

    def __init__(self, config: CourseModelDefinitionConfig) -> None:
        self.config = config                               # la configuration, au constructeur

    def define(self, model: Model[CourseProblemInstance]) -> None:   # les règles, sans données
        tasks = model.set(lambda s: s.tasks, key="id_")
        rooms = model.set(lambda s: s.resources, key="id_")
        recipes = model.set(lambda s: s.recipes, key="id_")
        start = model.var(
            "start", tasks, high=5 * 24,
            span=model.param(lambda task: recipes[task.recipe_id].scheduling_options[0].processing_time, tasks),
        )
        options = model.subsets(
            lambda task: [o.resource_id for o in recipes[task.recipe_id].scheduling_options], tasks, of=rooms
        )
        room = model.assign("room", tasks, rooms, among=options)
        model.cumulative(room, start, demand=1, capacity=model.param(lambda r: r.capacity, rooms), name="room_capacity")
```

La résolution, une fois le modèle écrit (`tests/optimization_service.py`) : tout se configure au
constructeur, et `solve` fait le reste sans nommer le problème.

```python
compiled = ModelCompiler(definition)                   # une fois ; cythonize=False : Python pur
precheck = InfeasibilityCheck(PrecheckConfig(up_to=PrecheckLevel.GROUP, time_budget_seconds=20))
service = OptimizationService(compiled, precheck, KernelSolver(KernelSolverConfig()), OutputMapper(CourseSolution))

result = service.solve(problem, warm_start, fixed, check=True)   # check : la pré-vérification tourne
result.precheck_issues                                 # ce qu'elle a signalé ; None si check=False
result.solution                                        # ModelSolution, en clés : ce qu'une résolution suivante reprend
result.output                                          # le schéma de sortie du problème, rempli
```

`solve` lie le problème **une fois** ; la pré-vérification, le solveur et la sortie lisent ce même modèle
lié (`BoundModel`). C'est pourquoi `KernelSolver` ne reçoit pas le compilateur au constructeur : lier est
l'étape coûteuse (≈ 2 s à 29 448 blocs), et un `BoundModel` ne naît que de `ModelCompiler.bind`, son
type suffit à garantir qu'il est lié.

**La sortie est une configuration.** Une affectation n'a qu'une forme, `VariableAssignment` (élément,
valeur, cible, par les clés que déclarent les ensembles) : elle voyage en JSON et se redonne en
`warm_start` ou `fixed`. Le problème déclare sa sortie, un simple `BaseModel` dont chaque champ dit où il lit
par son alias (`AliasPath("element", "group_key")`, `"value"`, `"end"`, `AliasPath("target", "id_")`),
sous `assigned`, une entrée par élément placé. `OutputMapper(schema).map(bound, solution)` résout les clés
sur le problème lié et remplit le schéma : aucun code propre au problème (≈ 140 ms à 1 262 blocs).
Aucune classe de base à hériter. Un champ lu par `AliasPath` demande `ConfigDict(validate_by_name=True)`
sur son modèle : sans lui, le greffon mypy de Pydantic refuse l'alias (`PlacedBlock` le pose). Le même
réglage permet de relire une sortie depuis son propre JSON, écrit par noms de champs.

**Une seule façon d'écrire chaque chose** (aucune surcharge `@overload`) :

| Écriture | Sens |
|---|---|
| `group.member`, `group.pair` | un membre libre, une paire libre : chaque accès en crée de nouveaux, qui parcourent tous les membres ou toutes les paires (propriétés) |
| `group.size` | le nombre de membres d'une collection |
| `model.count(x, condition)` | les membres `x` où la condition tient |
| `model.sum(x, valeur)`, `model.min(x, valeur)` | somme, minimum sur les membres |
| `model.largest(a, b, …)`, `model.smallest(a, b, …)` | le plus grand, le plus petit de quelques valeurs |
| `model.exists([...])`, `model.forall([...])` | au moins une, toutes vraies |
| `model.sort(group, key=start)` puis `order.previous`, `order.current` | deux membres consécutifs, dans l'ordre d'une décision |
| `param[x]` | un paramètre lu en un élément |
| `recipes[task.recipe_id]` dans une lambda | la jointure sur la clé d'un ensemble |
| `model.within(lambda task: task.time_windows, block)` | un membre de la liste lue sur un élément |
| `model.group_by(lambda task: task.teacher_id, tasks)` | un sous-ensemble par valeur de la clé |
| `for group in family`, `for group in assignment.groups` | les deux seules boucles dépliées : une instance par sous-ensemble ou par cible |
| `model.constraint(expr, name=, scope=group)`, `model.minimize(expr, name=, weight=)` | dure ; souple, pondérée |
| `sum([...])` (Python) | des expressions additionnées |

**Garde-fous.** Une expression refuse d'être évaluée comme vrai ou faux (`if`, `and`, `in` sur une
expression lèvent `TypeError`). Un second `m.var` ou `m.assign` est refusé : le noyau n'en tient qu'un de
chaque.

## 1 ter. Proposition (2026-10-27, **refusée par F-A**) : écrire le modèle sur le schéma typé

**Refusée** : aller chercher `data.recipes_by_id[task.recipe_id]` dans une lambda, juste pour obtenir une
durée, est très bizarre ; et l'exemple nommait `CourseModel` ce qui est une définition
(`CourseModelDefinition`). Le texte est gardé pour mémoire.

F-A refuse `m.schema.tasks.duration`, que l'IDE ne reconnaît pas. Il refuse aussi les deux formes en
lambda proposées : il faudrait déclarer un ensemble avec sa clé (`key="id_"`) et un paramètre par
champ, alors que le schéma les donne déjà. Proposition :

- `define(self, m, data)` reçoit `data` **typé comme le schéma** (`CourseProblemInstance`). L'IDE et
  mypy vérifient chaque attribut. À l'exécution, `data` est un enregistreur qui note les chemins.
- **Les éléments sont les objets des listes du schéma**, repérés par leur position : `for task in
  data.tasks:` veut dire « pour chaque tâche » (une seule passe symbolique, `task` typé `CourseBlock`).
  Aucun `m.set`.
- **Lire un champ, c'est un paramètre** : `task.time_windows`, `task.teacher_id`. Aucun `m.param`.
- **Les liens passent par les dictionnaires que le schéma offre déjà**, sans clé déclarée :
  `data.recipes_by_id[task.recipe_id]` rend l'objet `Recipe` lui-même. À la liaison, l'objet obtenu est
  reconnu par identité dans `data.recipes`.
- **Les groupes viennent aussi du schéma** : `for team in data.blocks_by_teacher.values():` (une
  propriété du schéma, s'il l'a), puis `for a, b in m.pairs(team):`. Aucun `group_by`.
- **Les décisions** : `start = m.interval("start", data.tasks, length=..., high=...)` et
  `room = m.assign("room", data.tasks, to=data.resources, among=...)`.
- `m.constraint(expr, name=)` est dure. `m.minimize(expr, name=, weight=)` est souple, avec un poids.

```python
class CourseModel(ModelDefinition[CourseProblemInstance]):
    def define(self, m: Model, data: CourseProblemInstance) -> None:
        start = m.interval(
            "start", data.tasks, high=5 * 24,
            length=lambda task: data.recipes_by_id[task.recipe_id].scheduling_options[0].processing_time,
        )
        room = m.assign("room", data.tasks, to=data.resources,
                        among=lambda task: data.rooms_by_task[task.id_])
        for task in data.tasks:
            m.constraint(
                m.any_of((w.start <= start[task]) & (start.end(task) <= w.end) for w in task.time_windows),
                name="window_coverage",
            )
        m.no_overlap(room, start, name="no_room_overlap")
```

**Limites à accepter.**
- Python ne laisse pas surcharger `and`, `or`, `if` ni `max()` : on écrit `&`, `|`,
  `m.if_then_else`, `m.max`, comme aujourd'hui.
- Une compréhension de liste sur une liste symbolique ne fait qu'une passe. Seuls `m.any_of`,
  `m.count_over`, `m.sum_over` et `m.min_over` la consomment.
- Un lien qui n'existe pas dans le schéma (par exemple `rooms_by_task`) s'y ajoute comme propriété, ou
  s'écrit dans la lambda.

## 1 bis. Le catalogue de contraintes : générique seulement

| Méthode | Sens |
|---|---|
| `model.no_overlap(groups, interval, name=, level=)` | deux intervalles d'un même groupe ne se chevauchent pas ; `groups` : une famille, une affectation ou une collection |
| `model.cumulative(groups, interval, demand, capacity, name=, level=)` | à tout moment, la somme des `demand` des intervalles en cours d'un groupe reste sous `capacity` (un nombre, ou une capacité par cible d'une affectation) |

Ces deux méthodes du modèle posent leurs expressions et refusent une décision sans longueur ou une
capacité par cible hors d'une affectation (`ValueError`). Une contrainte n'entre au catalogue que
si c'est une primitive des outils de référence (CP-SAT
`AddNoOverlap`/`AddCumulative`, MiniZinc `disjunctive`/`cumulative`). Une règle propre à un problème
s'écrit en expressions ordinaires dans le modèle de ce problème : l'espacement des blocs d'un cours
(`PartSpacing`, retiré du catalogue le 2026-10-26) est maintenant dans `tests/course_model.py`.

## 2. Lire le schéma

| Déclaration | Sens |
|---|---|
| `m.set(lambda s: s.tasks, key="id_")` | les éléments d'une liste ; `key` est le champ qui les nomme (texte ou entier : une clé voyage en texte) |
| `m.param(lambda task: task.recipe_id, tasks, of=recipes)` | un texte (ou un objet) qui nomme un élément de `recipes` devient sa position ; un nom inconnu est refusé |
| `m.param(lambda task: task.course_id, tasks)` | un texte sans `of=` est une catégorie : seule l'égalité compte |
| `m.set(..., distinct=True)` | des objets égaux (copies) comptent une seule fois ; deux ensembles sur la même liste, l'un `distinct`, l'autre non, se lisent chacun à part |
| `m.subsets(lambda task: [...], tasks, of=rooms)` | par élément, un sous-ensemble de `of` |
| `m.assign(name, tasks, rooms, among=options)` | le domaine déclaré : les cibles que chaque élément peut prendre |
| `m.group_by(lambda task: task.teacher_id, tasks, same_order_as=)` | un sous-ensemble par valeur de la clé (une liste de clés : l'élément rejoint chacune) ; `same_order_as` suit l'ordre d'un autre regroupement |
| `m.lookup_table([...])[kind[element]]` | une valeur donnée dans le modèle, lue à une position |
| `items[0]` dans une lambda | un élément d'une liste par position |

Les `RootModel` qui enveloppent une liste, les propriétés et les `cached_property` se lisent comme
des champs. Un chemin faux est refusé au moment où le modèle est écrit (`RecordedPath`).

## 3. Le noyau : un seul code, compilé ou en Python pur

`solver/kernel/kernel.py` est écrit en Cython « mode Python pur », comme
`service/course_assignment/alns/compiled_constraints.py`. Pour un modèle, `KernelSource` écrit le texte de
`kernel.py` suivi d'une sous-classe `ModelKernel` générée (ses tableaux et cinq blocs : `_around`, `_full`,
`_around_set`, `_constants`, `_violations`). `KernelBuilder` compile ce texte avec Cython (une fois par
source, en cache disque) ou l'importe tel quel en Python pur (`ModelCompiler(definition, cythonize=False)`).
`Kernel.COMPILED` dit lequel tourne. Le texte dépend du modèle seul, jamais des données.

Il n'y a plus d'interpréteur : le noyau compte les règles sans décision (`_constants`), rend les
violations en mots (`kernel.violations`, avec les objets du schéma), et le même code tourné en Python pur
sert de témoin au code compilé. Les références **indépendantes** restent les contraintes écrites à la main
du projet (`scripts/compare_course_definition.py`) et le compte direct du problème de livraisons.

`cdivision=False` : `//` arrondit vers le bas, compilé ou non.

## 4. Pré-vérification (`solver/infeasibility/precheck.py`) : trois niveaux, délai réglable

`InfeasibilityCheck(PrecheckConfig(up_to=..., time_budget_seconds=...)).run(bound)` donne des
`PrecheckIssue` : ce que les règles dures rendent impossible, avant toute recherche. Chaque constat nomme les règles qui
**suffisent** à le prouver.

**Une exécution ne dépasse pas `PrecheckConfig.time_budget_seconds`, s'il est réglé.** Sans
réglage, il n'y a pas de délai : l'appelant accepte d'attendre que chaque recherche atteigne son
plafond (`group_budget`, `witness_budget`). Le délai est un `MaxTime` du backend, armé au début de
`run`. Il est consulté entre les éléments du niveau 1, entre les groupes, dans la recherche du noyau
(toutes les 4 096 évaluations, `MaxTime.CLOCK_STRIDE`) et dans les explications. Ce qui n'est pas
décidé à temps n'est pas prouvé. La liaison se fait avant `run` : elle ne compte pas.

**Ce qui lie des éléments, pour tout modèle.** Chaque gabarit publie, par instance :
- `members`, le domaine de ses membres libres ;
- `reads`, ce que lit chaque évaluation en plus de ses membres libres.

Une instance de paire lie ses membres. Une instance de membre lie ses membres à ce qu'elle lit. Toute
instance lie ce qu'elle lit. Une règle n'est lue autour d'un élément qu'une fois placés tous les
éléments que lisent ses instances qui le touchent. Un ensemble décidé compte pour tous les éléments qui
peuvent y aller. Aucune lecture partielle n'est donc fausse, et aucune hypothèse de monotonie n'est
faite.

**Jusqu'où aller : `PrecheckConfig.up_to`**, un `PrecheckLevel`. Chaque niveau fait tourner ceux d'avant,
du moins coûteux au plus coûteux ; `COMPONENT` (le défaut) les fait tous.

| `up_to` | Ce que la pré-vérification cherche |
|---|---|
| `PrecheckLevel.ELEMENT` | chaque élément seul, et la charge de chaque ressource |
| `PrecheckLevel.GROUP` | puis chaque petit groupe qu'une règle dure lie (`group_limit`) |
| `PrecheckLevel.COMPONENT` | puis chaque composante, de toute taille |

**Les trois niveaux :**
1. **Élément sans placement** : ses débuts et ses ensembles possibles, filtrés par les règles qui ne
   lisent que l'un d'eux et ne lisent que lui. Tous les débuts d'un élément sont essayés en un seul
   appel compilé (`Kernel.lone_placements`).
2. **Groupe qu'une règle lie** (≤ `group_limit` éléments).
3. **Composante** : les éléments que toutes les règles dures lient de proche en proche, de toute
   taille, la plus petite d'abord.

**Chaque groupe et chaque composante passent par trois étapes :**
1. Une recherche en profondeur qui lit tout dès qu'un élément est placé. Elle ne fait que guider : un
   placement trouvé est vérifié exactement.
2. Si elle cale, une recherche locale (`Kernel.min_conflicts` : moindres conflits, un coup sur dix au hasard,
   tirage fixe). Un placement trouvé suffit : il n'y a rien à prouver. Aucune de ces deux étapes ne
   prouve l'impossibilité.
3. Sinon, la preuve exacte (`Kernel.depth_first`) : en profondeur, avec élagage en avant. Chaque placement
   retire aux éléments suivants les candidats qu'il interdit, et une liste vide fait reculer.

**Explication** : on part des seules règles qui ont rejeté quelque chose pendant la preuve, puisque
les autres n'ont pas agi. On retire chacune à tour de rôle, avec le budget de la preuve ; un essai non
décidé garde la règle. Passé l'échéance, on garde le reste. Le résultat reste sûr.

**Résultats (2026-10-26, noyau compilé, mesurés par l'auteur) :**

| Cas | Temps | Résultat |
|---|---|---|
| 3 blocs à jours consécutifs sur 2 jours ouverts (`precheck_test.py`, 2026-10-30) | 0,02 s | groupe prouvé |
| 2 blocs séparés d'un jour sur 2 jours ouverts (`precheck_test.py`, 2026-10-30) | 0,01 s | groupe prouvé |
| chaîne d'une tâche d'atelier (`job_shop_model.py`) | 4 ms | groupe prouvé : `time_window`, `job_order` (même code que les cours) |
| 13 blocs identiques à débuts distincts en 12 h | immédiat (budget épuisé sans l'ordre des échangeables) | composante prouvée ; 12 blocs : rien |
| équipe et co-début (3 blocs) | court | composante prouvée au niveau 3 |
| 3 656 blocs (`LargeCourseProblem(600)`) | 0,8 s (liaison 0,2 s) | rien ; 1 877 groupes et 118 composantes, tous placés |
| 29 448 blocs (`LargeCourseProblem(4900)`) | 7,5 s (liaison 1,9 s, niveau 1 2,4 s, niveau 2 2,1 s, niveau 3 1,4 s) | rien ; 15 085 groupes et 906 composantes, tous placés |

**Les essais suivent les données** (F-A, 2026-10-30). Dans v13, un groupe-cours compte au plus trois
blocs, et un groupe à jours consécutifs deux ou trois. Les anciens cas de quatre et six blocs
n'existent pas : ils sont retirés, l'essai du délai avec eux. Six blocs à jours consécutifs restent
non prouvés (une instance sur six blocs, 45⁶ placements), mais une semaine de cinq jours ne les tient
pas, et `BlockSpacingFeasibilityCheck` du backend le signale dès la lecture du fichier.

**Ramasse-miettes.** La liaison suspend le ramasse-miettes. La première allocation qui suit paie la
collecte différée, ≈ 1,3 s à 29 448 blocs, comptée dans le niveau 1. `gc.freeze()` après la liaison
l'éviterait, mais il agit sur tout le processus : **décision de F-A**, non prise.

### 4.1 Demande suivante (2026-10-26) : un niveau 3 générique, 20 s fermes, une liaison plus rapide

F-A demande un **niveau 3** meilleur et générique que l'ancien (retiré), libre dans sa forme : du
déterministe, de la recherche locale, ou un mélange. Le plafond devient **20 s, toujours, jamais plus**,
pour une exécution complète. Ce qui n'est pas décidé à temps reste « non prouvé » et n'est pas une
erreur. **La liaison doit aussi aller plus vite.**

Ce qui est confié à deux agents `@dev`, chacun dans des fichiers différents :

| Agent | Portée | Fichiers | Critère |
|---|---|---|---|
| Niveau 3 | les groupes liés de toute taille (la chaîne d'un atelier, six blocs à des jours consécutifs) ; un délai global de 20 s qui borne la preuve **et** l'explication | `solver/infeasibility/`, `solver/kernel/kernel.py` (ajouts seulement), essais de la pré-vérification | le cas des 4 blocs passe sous 20 s sans `xfail` ; l'atelier et les six jours sont prouvés ; rien de propre aux cours dans `modeling/` |
| Liaison | liaison et calcul des domaines du niveau 1 à grande échelle | `solver/binder.py`, `solver/compiled_model.py`, `solver/schema_reader.py`, `solver/kernel/shape.py`, `solver/kernel/codegen.py` | mêmes comptes et variations qu'avant ; 3 656 et 29 448 blocs nettement plus rapides, mesurés avant et après |

**Critère éliminatoire (F-A) : si ce n'est pas générique pour n'importe quel problème, c'est un
échec.** Le niveau 3 ne raisonne que sur ce que le moteur connaît de tout modèle : les éléments, leurs
domaines de début et de cible, les instances des règles dures et l'évaluation du noyau. Il ne connaît ni
jours, ni blocs, ni salles. L'atelier et les cours sont prouvés par le même code. Aucun cas d'essai
n'est réglé en ajustant ses données ou ses budgets. Une preuve qui ne tient que par un détail propre à
un problème est abandonnée et notée « non prouvée ». Les accélérations de la liaison suivent la
structure des expressions, jamais le modèle des cours.

Les deux agents respectent les règles de la section 6. L'auteur relit et mesure le résultat avant de
l'accepter.

**Premier tour (mesuré par l'auteur)** :
- liaison : 1,73 s → 0,18 s à 3 656 blocs ; 13,9 s → 1,72 s à 29 448 blocs ;
- 4 blocs prouvés en 3,7 s (40 s avant) ; chaîne de l'atelier prouvée en 2 ms ;
- exécution complète à 29 448 blocs : 10,6 s, surtout le niveau 1 ;
- six jours : **non prouvés** (une seule instance sur six blocs, 45⁶ placements, rien n'élague).

**Remue-méninges des deux agents et de l'auteur (demande de F-A), puis retour au travail** :

| # | Idée | Pourquoi c'est générique | Décision |
|---|---|---|---|
| 3 | `Template` publie, par instance, les éléments qu'elle lit (collections fixes en clair ; un ensemble décidé ou une famille lue à un indice calculé : « tout membre de… ») | mêmes règles structurelles que la liaison, par classe de nœud | **à faire** (liaison) ; la pré-vérification en tire un graphe exact et retire l'hypothèse « ajouter des éléments ne répare pas un agrégat » |
| 6 | Éléments interchangeables (mêmes domaines, mêmes valeurs des paramètres que lisent les règles dures, mêmes sous-ensembles et collections, aucune référence fixe à l'un sans l'autre) : on fixe leur ordre, dans l'ordre de la clé de tri | toute solution permutée reste une solution, quel que soit le modèle ; les paires consécutives d'une suite deviennent définitives, donc élaguent | **à faire** : la liaison fournit les paramètres lus et l'appartenance ; la recherche du noyau impose l'ordre. Piège : `SortedList` départage par position |
| 7 | La boucle des débuts du niveau 1 dans une seule méthode compilée du noyau | aucune lecture d'expression | **à faire** (pré-vérification) |
| 8 | Domaines du niveau 1 mémorisés par signature d'élément (ses lignes dans les gabarits qui ne touchent que lui, longueur, `among`) | signature tirée des gabarits, pas du modèle | **à faire** : signature par la liaison, cache par la pré-vérification |
| 1 | Noyau d'intervalles : évaluation min/max de chaque instance sous des domaines partiels | sûr par classe de nœud | **plus tard** : un second chemin pour chaque nœud dans `codegen.py`, coûteux |
| 5 | Classes de valeurs (un début lu seulement par `//K` ou des comparaisons fixes) | traçage du code généré | **plus tard** : inutile dès qu'une règle de chevauchement lit le début |
| 4 | Impasses apprises et retour arrière non chronologique | seulement les comptes du noyau | **plus tard**, si 6 ne suffit pas |
| 2 | Domaines du niveau 1 vectorisés par gabarit | — | **rejeté** : un interpréteur (règle 10) ; 7 donne le même gain |

Les deux agents s'accordent sur l'interface entre eux. Ils gardent les critères : générique ou échec,
20 s au plus, mêmes comptes qu'avant.

**Second tour : ce qui est fait (mesuré par l'auteur)**

| # | Résultat |
|---|---|
| 3 | **Fait.** `Template.members`/`member_offsets` et `reads`/`read_offsets`. Une entrée `e < 0` vaut « tout l'ensemble décidé `-1-e` ». La pré-vérification en tire son graphe et le moment où une règle peut être lue. L'hypothèse de monotonie est retirée. |
| 6 | **Fait.** `ModelCompiler.element_classes` (`solver/element_classes.py`) et l'ordre imposé dans `Kernel.search`. Aucun gain sur les cours, où chaque élément est seul dans sa classe. Gardé parce qu'il prouve des cas qu'on ne prouve pas sans lui (13 blocs identiques en 12 h : budget épuisé sans l'ordre). Voir les contre-exemples plus bas. |
| 7 | **Fait.** `Kernel.lone_placements` : les domaines du niveau 1 coûtent 0,33 s à 29 448 blocs. |
| 8 | **Abandonné d'un commun accord.** Chaque élément des cours avait sa propre signature, et la calculer coûtait 1 à 2 s pour un gain d'au plus 0,33 s. |
| + | **Recherche locale du témoin** (`Kernel.repair`). Elle est devenue nécessaire avec les lectures exactes : sans elle, 36 composantes réalisables allaient jusqu'au délai à 29 448 blocs. |
| + | **Défaut corrigé dans `around` et `delta`** : un agrégat sur une autre collection, dans un terme à membre ou à paire, n'était pas réévalué. Sur `cumulative` (`SharedRoomDefinition`), 69 variations sur 300 étaient fausses avant la correction, 0 après. Un essai écrit d'abord, en échec, fixe le cas, compilé et en Python pur. |

**Contre-exemples débattus, gardés comme essais** : deux éléments ne sont échangeables que si rien
ne les distingue.
- **Une liste triée mêlant des classes** : les égalités de début se tranchent par la position, donc
  échanger deux éléments égaux réordonne la liste. Une liste ne garde ses membres échangeables que
  s'ils sont tous d'une même classe.
- **Une position lue comme valeur** (`start[a] >= a`, `rival < block`) : tous les éléments de cette
  collection restent seuls.
- **Une paire lue de façon asymétrique** (`at[first] < at[second]`) : ses deux éléments restent
  séparés.

### 4.2 Ce que F-A veut vraiment (2026-10-26) : ses contrôles, mais génériques

> « Ce que je veux, c'est pré-vérifier qu'une personne peut entrer, compte tenu de toutes les données
> pertinentes. Avec 1 000 personnes à placer, on ne va pas optimiser pour prouver la faisabilité. Je
> veux le même genre de contrôles que les miens, mais automatiques, génériques. »

Les contrôles de F-A (`apps/backend/.../ingestion/validation/checks.py`) sont des **conditions
nécessaires, entité par entité, sans recherche**. Chacun compare une demande à une offre et nomme
l'entité fautive avec ses chiffres :
- `TeacherLoadCheck` : heures de cours d'un enseignant contre ses heures plaçables, découpées en blocs
  entiers (`BlockFit`) ;
- `RoomGroupLoadCheck` : heures demandées à un groupe de locaux contre les heures qu'il offre ;
- `RoomTypeFeasibilityCheck` et `CapacityCheck` : aucun local admissible, ou aucun local assez grand.

**Ce que ça veut dire pour le moteur.** Les niveaux 2 et 3 actuels cherchent un placement (une
recherche) : c'est coûteux et c'est la mauvaise direction pour 1 000 personnes. Il faut plutôt des
**bornes de charge génériques**, linéaires, par entité. Elles se tirent des primitives que le modèle
déclare déjà (`no_overlap`, `cumulative`, l'affectation `among`) et des domaines du niveau 1, sans
rien savoir des cours :
- **par groupe d'une ressource** (les blocs d'un enseignant, d'un local, d'un étudiant) : somme des
  durées × demandes ≤ ce que la capacité offre sur les heures où ces éléments peuvent être, découpées
  en morceaux entiers comme `BlockFit` ;
- **par ensemble de cibles** (un groupe de locaux) : la demande des éléments qui ne peuvent aller que
  dans ces cibles ≤ ce que ces cibles offrent (condition de Hall) ;
- **par élément** : le niveau 1 actuel (aucun début, aucune cible) ;
- **sans temps** (affectation seule) : la même borne, avec la capacité seule.

Chaque constat nomme l'entité (l'étiquette du groupe : l'enseignant, le local) avec la demande et
l'offre, comme les messages de F-A. Le temps est linéaire : 1 000 ou 30 000 éléments, sans recherche.

**Décision de F-A : les niveaux 2 et 3 de recherche sont retirés.** La pré-vérification devient le
niveau 1 et les bornes de charge par entité, sans aucune recherche. Ce qui part :
- `Kernel.search`, `Kernel.repair` et l'ordre des échangeables ;
- `solver/element_classes.py` : 495 lignes, jugé non sûr par la revue B (trois contre-exemples) ;
- les preuves par recherche : 4 blocs, chaîne de l'atelier, 13 blocs identiques.

Ce qui reste du second tour :
- `members`/`reads` et la correction de `around`/`delta`, qui sont la justesse du noyau ;
- `lone_placements` ;
- l'échéance de 20 s ;
- la liaison rapide.

Les constats de la revue B sur la justesse du noyau sont à corriger :
- 1 : déduplication `_stamped` ;
- 3 : un élément nommé par des données (`at[partner[e]]`) est refusé par `UnsupportedError` tant
  qu'il n'est pas suivi exactement, et un bloc vide n'est jamais généré ;
- 9 et 10 : noms.

**Ordre exigé par F-A : rien n'est retiré avant la parité.** Le code des niveaux 2 et 3 (recherche,
témoins, liens de familles, échangeables) reste en place tant que le tableau de parité ne montre pas
qu'aucun contrôle de faisabilité de F-A n'en a besoin (par exemple `CoTeachingAvailabilityCheck`, qui
peut demander de placer ensemble les blocs de deux enseignants). Ce qui reste est justifié par le
contrôle qu'il sert.

**Critère d'acceptation (F-A) : faire exactement la même chose que mes contrôles.** Le niveau 1 et
les bornes de charge doivent trouver, sur les mêmes données, les mêmes problèmes que les contrôles de
faisabilité de F-A, entité par entité, avec les mêmes chiffres. Un script de parité est exigé
(`scripts/compare_checks.py`). Il fait tourner les contrôles de F-A sur le fichier maître et la
pré-vérification générique sur l'instance assemblée du même fichier, puis compare les entités signalées
et leurs demandes et offres. Toute différence est expliquée par une donnée que le modèle n'a pas (par
exemple, F-A reste optimiste pour un enseignant de plusieurs départements), sinon elle est corrigée. Les
contrôles de qualité des données (doublons, références inconnues, colonnes) restent hors du moteur :
ils ne disent rien sur la faisabilité.

**Parité, premier tour (mesurée par l'auteur sur `gabarit_CMV_FICHIER_MAITRE_v13_CONSOLIDE.xlsx`,
1 336 blocs, 3,8 s)** :
- enseignants : même verdict 306/307, mêmes demandes 307/307, mêmes offres 145/307 ;
- groupes de locaux : même verdict 45/56, mêmes demandes 44/54, mêmes offres 15/54 ;
- espacement impossible (`571-586-MV`) et deuxième local (`510-351-MV`) : trouvés par les niveaux 2
  et 3. Ces trois contrôles de F-A justifient de garder la recherche sur les groupes.

Écarts à fermer :
1. **Battement de labo** (3 groupes de labo que F-A signale surchargés et le moteur non, par exemple
   E-109 à 52 contre 38). À faire : un temps de transition déclaré sur `no_overlap`, comme les
   distances de transition de CP Optimizer. Le modèle d'essai le déclare, et la borne compte le
   battement entre deux morceaux d'une même plage.
2. **Dîner** (l'offre d'un enseignant est trop haute d'au plus 1 h par jour). À faire : des pauses
   déclarées sur `no_overlap`, comme les pauses des véhicules d'OR-Tools (`SetBreakIntervalsOfVehicle`).
   Le modèle d'essai y remplace sa règle du dîner par ces pauses, avec des comptes inchangés.
3. **Cibles que personne ne peut prendre** (H-114, CH-024 : F-A les dit sous-utilisées à 0). À faire :
   les signaler avec une demande nulle.
4. **Offre plus serrée** : le moteur compte les plages que les blocs peuvent occuper, F-A la semaine de
   la ressource. Trois groupes sont prouvés surchargés par le moteur et seulement « presque pleins »
   chez F-A (CH-103 45/44, GRP_351 159/156, GRP_388 134/132). **Décision de F-A.**
5. **Deuxième local** : F-A signale 11 cours, le moteur 1. Les 10 autres n'ont aucun groupe-cours cette
   session, donc aucun bloc dans l'instance. F-A contrôle la ligne du cours ; le moteur ne voit que ce
   qui est à placer.

**Parité, second tour (mesurée par l'auteur, même fichier, 1,8 s)** :

| Mesure | Même verdict | Même demande | Même offre | Avant |
|---|---|---|---|---|
| Enseignants (`TeacherLoadCheck`) | 306/307 | 307/307 | 294/307 | 306 / 307 / 145 |
| Groupes de locaux (`RoomGroupLoadCheck`) | 53/56 | 46/56 | 15/56 | 45 / 44 / 15 |

Avec l'enseignant surchargé à la main (10874), F-A et le moteur donnent tous deux 52/44, surchargé.

Ajouté, générique :
- `no_overlap(..., where=, transition=Transition(kind, same, other), breaks=Breaks(name, period,
  window, length))`, sur le modèle des distances de transition de CP Optimizer et des pauses de
  véhicules d'OR-Tools ;
- `bin_packing` (pour les problèmes sans temps) ;
- `Model.resources`, que la pré-vérification lit.

Le modèle d'essai déclare ainsi son battement de labo et son dîner, avec des comptes et des variations
identiques à ses anciennes règles (1 336, 3 656 et 363 éléments, compilé et Python pur). Une cible
qu'aucun élément ne peut prendre est signalée sous-utilisée (demande 0).

**Décision de F-A : l'offre plus serrée est gardée.** Le moteur compte les plages que les éléments
peuvent occuper, et c'est une preuve sûre. CH-103 (45/44), GRP_351 (159/156) et GRP_388 (134/132) sont
donc prouvés surchargés, là où le contrôle de F-A, plus optimiste, dit « presque plein ». L'enseignant
06187 (un bloc dans une plage de 3 h) est « presque plein » pour le moteur et non pour F-A, pour la même
raison.

**Vérification exacte de l'offre plus serrée (2026-10-27, à la demande de F-A).** Chaque groupe de
locaux que le moteur prouve surchargé a été résolu exactement avec CP-SAT, en modèle indexé par heure.
Les débuts possibles sont refaits à partir des plages brutes des tâches, pas du moteur, et seules deux
règles sont gardées : les plages et un seul bloc par local à la fois. Même avec ces débuts bruts, plus
larges, **les six sont impossibles** (`scratchpad/pool_exact/exact2.py`) :

| Groupe | Moteur (demande/offre) | F-A | Exact |
|---|---|---|---|
| CH-103 | 45/44 | 45/48 presque plein | impossible |
| GRP_351 (D-005, F-016, F-029, G-022) | 159/156 | 159/168 presque plein | impossible |
| GRP_388 (F-202, F-203, F-207) | 134/132 | 134/144 presque plein | impossible |
| CH-104 | 59/46 | 59/48 surchargé | impossible |
| CH-104, CH-147 | 97/92 | 97/96 surchargé | impossible |
| GRP_200_322_340_351_388_600 (8 locaux) | 422/384 | 422/384 surchargé | impossible |

L'offre plus serrée est donc la bonne : le contrôle de F-A dit « presque plein » pour trois groupes
réellement impossibles, parce qu'il compte la semaine entière du local. Le moteur ne compte que les
heures où ces blocs peuvent tenir, en morceaux entiers.

**Écart restant expliqué : le battement de labo compté par plage, non par jour.** Le battement du
mercredi coupe la journée en deux plages, et deux labos de part et d'autre attendent déjà hors des heures
offertes. Le moteur compte 50/46 pour E-109, F-A 52/48 ; le verdict est le même (surchargé). Le compte
par jour de F-A peut signaler un groupe réalisable, celui du moteur jamais.

## 5. Ce qui est vérifié

| Vérification | Script | Résultat |
|---|---|---|
| Les 13 règles contre les contraintes écrites à la main du projet, sur les vrais `CourseProblemInstance` | `scripts/compare_course_definition.py` (`tests/parity/definition_parity.py`) | 2026-10-28, backend du jour : 11 fichiers maîtres × 10 horaires, 0 différence sur 12 règles ; `teacher_room_changes` ±1 sur 2 horaires, quand un enseignant a deux blocs au même début (l'ordre des ex æquo diffère) |
| Les contrôles de charge de F-A contre la pré-vérification, entité par entité | `scripts/compare_checks.py` | 2026-10-28, fichier maître du jour : enseignants 309/310 même note (04909 : prouvé au niveau 2 ici), pools 56/58 (voir journal, point 51) |
| La pré-vérification en parallèle contre la version séquentielle | banc d'essai ponctuel | 2026-10-28 : constats identiques octet pour octet (fichier maître du jour × 3, cours générés de 60 et 600 enseignants × 3 germes compilés, 60 en Python pur, atelier) |
| La refactorisation contre le moteur d'avant (`HEAD`), problèmes générés de 60 et 600 enseignants | banc d'essai ponctuel | mêmes comptes sur chaque placement, mêmes variations sur 40 000 déplacements (règle d'espacement comprise, réécrite en expressions) |
| Le noyau compilé contre le même noyau en Python pur, éléments absents au hasard | `tests/solver/binding/model_compiler_test.py` (`blocks_left_out`, sur `LargeCourseProblem` : labos, co-départs) | 0 différence |
| Un schéma sans rapport avec les cours | `tests/solver/binding/model_compiler_test.py` (`TestDeliveries`) | 1 réseau × 30 plans, 0 différence |
| La pré-vérification sur un atelier | `tests/solver/infeasibility/precheck_test.py` | faisable : rien de prouvé ; niveau 1 : prouvé ; niveau 2 : la chaîne prouvée en `group` |

**Vitesse après la refactorisation** (3 656 blocs, meilleur de 3 à 5 essais, un cœur) : liaison 1,7 s
(1,7 à 2,0 s avant) ; `delta_place` ≈ 2,1 µs contre 1,9 à 2,0 µs avant (5 à 10 % plus lent).

**Liaison après le 2026-10-26 (suite)** (noyau compilé, meilleur de 3 à 5 essais) : 3 656 blocs,
1,73 s → 0,2 s ; 29 448 blocs, 13,9 s → 1,8 s.
- Chaque énoncé est lié une seule fois. Ses boucles deviennent des colonnes numpy, une valeur par
  instance, et la sonde note la valeur de chacun de ses nombres et de ses collections par instance.
- La forme est tirée de la sonde seule.
- Les listes d'instances par élément sont construites avec numpy à partir de `members` et `reads`.
- Le ramasse-miettes est suspendu pendant la liaison.
- Ce qui reste à 29 448 blocs est surtout la lecture du schéma (`SchemaReader`).

**Équivalence** : vérifiée contre le moteur d'avant, compilé et en Python pur.
- Cours à 60 et 600 enseignants : mêmes comptes, violations et variations (18 000 déplacements par
  taille).
- Livraisons, ateliers, `SharedRoom`, `DistinctStart` et un modèle de cas limites : mêmes comptes et
  violations. Les variations égalent un recompte complet partout ; seules celles de `SharedRoom`
  changent, parce qu'elles étaient fausses.
- Le texte généré ne dépend que du modèle.

## 6. Règles (contrat de revue)

Ces règles, posées par F-A, servent de grille à toute revue. Une revue cite leur numéro.

1. **Constructeur = configuration, méthodes = données.** Le constructeur reçoit ce qui est donné une
   fois ; une méthode reçoit les données qu'elle traite et peut changer l'état de l'objet. Un argument
   propre à une seule méthode y reste (`weights` de `bind`, `budget` de `search`). Un drapeau qui
   ne vaut que pendant un appel est un argument, pas un attribut. Les enregistrements (nœuds,
   déclarations, `Term`, `PrecheckIssue`, `TermShape`, `Template`) reçoivent leurs valeurs au constructeur.
   Exemples : `ModelCompiler(definition, cythonize=)` puis `bind(problem, lex_base)` ;
   `Binder(model)` puis `expand(data)` ; `InfeasibilityCheck(config)` puis `run(bound)`.
2. **Pas de classe sans configuration instanciée** (règle, pas une option). Une classe dont le
   constructeur n'aurait aucun argument ne s'instancie pas : elle n'a que des méthodes de classe ou
   statiques, comme `ProblemInstanceBuilder.build_course_instance` dans `apps/backend/src`. Si elle a
   besoin d'un état de travail pendant un appel, elle est fondue dans la classe configurée qui s'en sert
   (pas de `cls()` sur un constructeur vide).
3. **Méthode sans argument qui rend une valeur → propriété** quand c'est pertinent.
4. **Une seule façon d'écrire chaque chose** : pas de `@overload`, pas de signature « une chose ou une
   liste », pas de deux méthodes publiques pour le même effet.
5. **Pas de `Protocol`** (une `ABC` au besoin) ; pas d'abstraction pour une seule implémentation.
   Une `ABC` a **plusieurs vraies sous-classes** et une raison de les réunir (un comportement commun,
   un appelant qui les prend toutes) ; un faux d'essai ne compte pas comme sous-classe (2026-11-03,
   `CandidateEncoder` n'avait qu'un `BenchEncoder`).
6. **Ni variable globale ni fonction d'une ligne** (lambda, aide triviale) ; les constantes vivent dans
   leur classe. Seule exception : une lambda qui choisit un champ du schéma dans une déclaration
   (`m.param(lambda task: task.recipe_id, tasks)`), pour que l'IDE vérifie le chemin.
7. **Pas de classe inutile, pas de sur-ingénierie, pas de dette technique, pas de méthode qui n'a rien
   à faire dans sa classe** ; ce qui n'est ni utilisé ni testé est retiré. **Du code qui n'apporte rien
   est de la sur-ingénierie.** Une classe, une exception, un champ, un paramètre ou une enveloppe sans
   comportement ni information propre est retiré. Par exemple, une exception maison vide
   (`UnsupportedError`) cède la place à une exception standard (`ValueError`).
8. **Organisation** : les noms de dossiers de `apps/backend/src` (`problem_formulation`, `solver`) ; une
   classe vit là où elle sert ; pas de dossier `schema`, pas de dossier propre à un problème, pas
   d'OR-Tools, pas d'encodeur/décodeur (`SchemaPath` en tient lieu). Rien n'est migré dans
   `apps/backend/src`.
9. **Générique, sinon c'est un échec** : rien dans `modeling/` ne nomme un problème. Le catalogue ne garde
   que des primitives des outils de référence. Un besoin propre à un problème (dîner, battement)
   passe par une primitive générique déclarée (`Breaks`, `Transition`), jamais par du code qui le
   nomme. Les modèles de cours et d'étudiants ne servent qu'aux essais.
10. **Pas d'interpréteur** : le noyau est en Cython « mode Python pur », le même code compilé ou non.
11. **Pré-vérification** : trois niveaux, **le délai réglé par l'appelant, sans défaut** (F-A,
    2026-10-29, remplace « 20 s au plus, toujours ») ; ce qui n'est pas décidé à temps n'est pas
    prouvé. Si un cas n'est pas tenu, on le dit : aucun essai
    ni réglage ajusté pour masquer une lenteur. Non générique pour n'importe quel problème = échec.
12. **Commentaires rares et courts** (un ou deux lignes, une contrainte que le code ne montre pas), aucun
    séparateur de section ; une méthode qui prend des arguments s'écrit avec eux (`run(bound)`), ou
    sans parenthèses (`run`), jamais `run()` ; docstrings d'une ligne, une ligne par entrée
    `Args`/`Returns` (CLAUDE.md) ; lignes ≤ 99 caractères ; code et noms en anglais ; annotations de type partout.
13. **Revue** : un agent indépendant (`@dev`) vérifie ces règles contre le code ; l'auteur et les
    réviseurs doivent s'accorder, chacun ne cédant que devant un vrai argument. Une meilleure option est
    bienvenue si elle est justifiée.
14. **Chaque module, classe et méthode doit être nécessaire.** Une revue se demande pour chacun s'il sert
    vraiment (l'ancien `interpreter.py`, par exemple, ne servait pas : le noyau en Python pur le
    remplaçait). Pour s'inspirer, elle regarde comment F-A fait dans son code :
    `apps/backend/src/cegep_mv_schedule/service/course_assignment/alns/compiled_constraints.py` (noyau
    Cython en mode Python pur, sa repli en Python, son `.pyi`) et
    `apps/backend/src/cegep_mv_schedule/core/optimization/` (découpage `solver/`, `local_search/`,
    `objective/`). **Une classe, une responsabilité** : une classe qui en mêle deux (lire les plans et
    écrire les règles, par exemple) est coupée ; une classe d'une seule méthode sans état est une
    fonction. **Une seule façon de faire chaque chose** : une deuxième recherche, un deuxième chargement
    du noyau, une deuxième lecture des durées à côté de celles du moteur sont retirés au profit du
    moteur (2026-11-03, `mining/local_search.py` refaisait `KernelSolver`).

15. **Valider à la construction, pas à la lecture.** Une valeur est vérifiée une fois, quand l'objet est
    fait : par un `model_validator` Pydantic, ou par le constructeur d'une classe simple. Une propriété
    rend une valeur ; elle ne vérifie pas un type pour lever une erreur à chaque lecture (par exemple
    `FilteredSet.bound_mask`, qui vérifie qu'un masque est déjà un tableau NumPy).
16. **Sûr, sinon refusé.** Le moteur ne rend jamais un résultat faux en silence : une preuve
    (`PrecheckIssue` prouvé) doit être vraie, et une variation doit égaler un recompte complet. Ce que le
    moteur ne sait pas traiter exactement est refusé par une exception claire à la liaison (par exemple
    un élément nommé par des données, `at[partner[e]]`). Une borne qui ne tient pas dans un cas (une
    pause qui chevauche deux périodes) retombe sur une borne plus faible mais vraie.
17. **Même résultat après une optimisation ou une refactorisation.** Comptes, variations, violations et
    constats restent identiques. C'est vérifié par script contre la version d'avant, compilé et en
    Python pur, sur des modèles hors cours aussi. Un écart n'est accepté que s'il corrige un défaut, et
    il est alors nommé.
18. **Rien n'est retiré avant d'avoir vérifié qu'il ne sert pas** : à un contrôle de F-A, à une preuve,
    à un essai. La justification est écrite. Ce qui est retiré est copié de côté d'abord.
19. **Une référence à un outil extérieur est vraie.** Un docstring ou ce README ne cite une primitive
    (CP-SAT, CP Optimizer, OR-Tools, MiniZinc) qu'après avoir vérifié qu'elle existe et qu'elle a ce
    sens. `AddBinPacking` de CP-SAT, qui n'existe pas, a été cité à tort.
20. **Pas d'état de travail sur l'objet** : ce qui ne vaut que pendant un appel (la forme en cours de
    lecture, un drapeau, un curseur) est un argument ou une variable locale, pas un attribut.
21. **Le parallélisme n'utilise jamais tous les cœurs** : au plus `available_cores() - 2`, et seulement si
    un gain mesuré le justifie.

22. **Pas de classe préfixée d'un souligné.** Une représentation intermédiaire, comme l'état d'une
    exécution (`_Run`), vit dans le fichier de la classe qui s'en sert, sous un nom normal
    (`PrecheckRun`). Elle n'a pas de fichier à elle : elle fait partie de ce fichier.

23. **Au plus quatre arguments par méthode** (2026-10-28). Au-delà, les arguments qui vont ensemble
    forment un enregistrement (règle 22), ou la méthode fait trop de choses et se coupe.

24. **Des listes, pas des tuples** (2026-10-28, journal n° 42). Un tuple rendu ou reçu demande une vraie
    raison ; sinon une liste, un dictionnaire ou un enregistrement.

25. **Le moteur ignore les essais** (2026-11-02). Rien dans `modeling/` n'existe pour les essais : ni
    réglage, ni variable d'environnement, ni branche (par exemple un `-O0` pour compiler plus vite). Un
    essai lent se corrige dans `tests/` : Python pur par défaut, Cython seulement pour comparer les deux.

26. **Le code reçu d'ailleurs passe ces règles avant d'être installé** (2026-11-03). Un paquet livré
    dans un zip ou par une autre session est relu classe par classe, script par script ; ce qui ne
    passe pas est réécrit ou retiré, jamais seulement déplacé. `mining/` a été installé tel quel et
    portait une deuxième recherche, une `ABC` à une seule implémentation et des classes de banc dans
    `scripts/`.

27. **Un script est un point d'entrée court qui montre ce que les essais ne montrent pas** (une mesure,
    un banc, un usage complet) (2026-11-03). Ses classes vivent dans le paquet ou dans `tests/` (un
    banc est un modèle d'essai, comme `tests/course_model.py`) ; un script qui ne montre rien de neuf
    est retiré.

28. **Du code qui respire** (2026-11-03). Une ligne vide avant et après chaque bloc (`if`, `for`,
    `while`, `with`, `try`), sauf quand il ouvre ou ferme le bloc qui le contient ; une ligne vide
    entre les étapes d'une méthode. Un essai sépare préparation, action et vérification par une ligne
    vide.

29. **Les essais suivent l'arborescence du code** (2026-11-03). L'essai de
    `optimization/<paquet>/<module>.py` est `tests/<paquet>/<module>_test.py` ; les modèles et bancs
    d'essai d'un paquet vivent dans son dossier d'essais (`tests/mining/shop.py`).

## 6 bis. Revue du 2026-10-26 (réviseurs A et B, `@dev`) : décisions accordées

L'auteur a proposé une position par constat ; les deux réviseurs ont répondu ; voici ce sur quoi les
trois s'accordent. Rien n'a été codé avant cet accord ; tout est fait depuis, et les deux réviseurs
ont vérifié dans le code que chaque ligne l'est (dernier point : `KernelSource` n'atteint le gabarit
courant que par son curseur `self.template` ; ses aides sans argument restent des méthodes, car elles
écrivent du code et enregistrent des tableaux).

| # | Constat | Décision accordée | Règles |
|---|---|---|---|
| 1 | Le catalogue n'a plus de lecteur ; `GlobalConstraint` n'a qu'un enfant | retirer `catalog`, `_post` et les 4 classes ; `no_overlap`/`cumulative` posent leurs expressions et lèvent `UnsupportedError` (l'essai qui attendait `ValidationError` suit) ; `IntegerDecision.end` lève aussi `UnsupportedError` | 5, 7, 14 |
| 2 | `SchemaReader.load` crée puis remplit | `SchemaReader(root)` : l'objet lu est sa configuration ; son cache de lecture remplace `Binder.read` ; `Binder._params` reste au binder (composition `via`, tables du modèle) | 1, 2, 4 |
| 3 | `KernelSource` : constructeur vide ; le texte dépend des données | `KernelSource(templates, n_elements, n_sets)`, créé par la méthode de classe `write` ; `fixed_members` privé ; nombres de constantes et de collections lus sur le gabarit, son rang porté par `Template` ; `zero_cols` décidé par la sonde seule, **à condition** que la sonde écrive une boucle comme valeur inconnue (et non 0) : le texte ne dépend alors que du modèle | 1, 2, 14 |
| 4 | `Template` rempli par `add()` | construit avec toutes ses instances ; la vérification de `scope` passe au constructeur | 1 |
| 5 | `ForkPool` non mesuré | mesuré sur 16 cœurs : 1 contre 8, 2,1 / 2,3 s à 3 656 blocs, 7,4 / 7,3 s à 12 009 (la liaison domine) : retiré avec `workers` et ses essais ; **revenu depuis** : `ForkPool` et `PrecheckConfig.workers` cherchent de nouveau les groupes indécis | 14 |
| 6 | `cache_dir` jamais passé ; `around_set_count` interne ; `paused_collector`, `fixed_members` publics | retirés ou privés ; les essais de `paused_collector` appellent la méthode privée, exprès | 14 |
| 7 | Deux écritures d'une boucle | `IndexSet` ne s'itère plus ; un seul membre libre `group.member`, une seule paire `group.pair`, deux positions nommées `order.previous` et `order.current` (variante validée : le noyau ne lit que les décalages -1 et 0) ; les deux seules boucles dépliées (une instance par valeur) sont `for group in family` et `for group in assignment.groups` | 4 |
| 8 | `shapes` double les gabarits ; l'état d'une liaison reste sur l'objet | `templates` remplace `shapes`, documenté « de la dernière liaison » ; l'état reste (règle 1 : une méthode qui reçoit des données change l'état ; le noyau rendu possède ses tableaux ; la pré-vérification lie et lit dans le même appel) | 1 |
| 9 | Docstrings du noyau | `Args`/`Returns` partout, ligne `# cython:` coupée en deux, commentaire d'en-tête juste, `Template` « sonde » | 12 |
| 10 | Lambdas dans les essais | sans objet hors `fork_pool_test.py` (retiré) ; `{2, 4}.issubset` dans `precheck_test.py` | 6 |
| 11 | Fenêtres écrites trois fois | une seule méthode statique publique `post_window_coverage(model, tasks, start)` dans le modèle de cours, appelée par les deux modèles d'essai (aucun cycle d'import) | 7 |
| 12 | Carte des classes fausse | refaite après les changements | — |
| 13 | `count_over` et `sum_over` | gardés : compter n'est pas sommer, et le compte permet l'arrêt au premier trouvé | 4 |
| 14 | `cast("Kernel", …)` vise la copie du module généré | gardé : rien ne fait `isinstance` | — |
| 15 | Collections liées héritant de l'API d'écriture | gardé : réutiliser les nœuds à la liaison évite une seconde hiérarchie | 7 |


---

<!-- Extrait 2 : ancien docs/optimization-engine.md, lignes 1033 à 1674 (7.5 et 8) -->

### 7.5 Ce qui n'est utilisé par rien aujourd'hui

`Kernel.delta_place` ne sert plus à la recherche du moteur (elle passe par `relocate`), mais
`mining/local_search.py` l'appelle et trois essais la vérifient : gardée.

## 8. Demandes de F-A (journal)

Le plus récent en bas. ✅ fait, 📝 noté pour plus tard, ❌ non tenu.

### 2026-10-26 — refactorisation du moteur

1. ✅ **Méthodes sans argument** → propriétés quand c'est pertinent (`members`, `pairs`, `member`, `size`,
   `groups`, `positions`, `const_matrix`, `group_loops`, et côté noyau `full_counts`, `full_hard`,
   `full_soft`, `placement`, `violating_elements`, `violations`).
2. ✅ **Commentaires inutiles** supprimés, et tous les séparateurs de section.
3. ✅ **Pas d'interpréteur** : le noyau en Cython « mode Python pur », compilé ou non (section 3).
4. ✅ **Organisation** comme `apps/backend/src` ; une classe vit là où elle sert ; pas de dossier `schema`
   ni propre à l'ordonnancement, pas d'OR-Tools, pas d'encodeur/décodeur. Rien migré dans `apps/backend/src`.
5. ✅ **Classes inutiles** fondues dans celle qui s'en sert : `Compiler`, `PausedCollector`,
   `ViolationExplainer`, l'interpréteur, `template.py`, `KernelIntegers`, `Partition`, `SymbolIds`,
   `OpFamily`, `CountScope`, `Grouping` (dans `SubsetFamily`), `LoopSubset` (dans `Subset`),
   `SubsetIndices`, `ConstantArray` (des tableaux NumPy).
6. ✅ **Pas de surcharge** ; une seule forme par signature (`param(name, over, of)`, `group_by(*by, of=)`,
   `scope=collection`, `bind` seul, `insert` seul).
7. ✅ **Explication des violations** : les noms viennent du modèle, les objets du schéma lié.
8. ✅ **Pas de `Protocol`** ; `ModelDefinition` est une ABC.
9. ✅ **Carte des classes** : section 7.
10. ✅ **`op.py`** : un nœud par sorte (opérandes, opérateurs), opérateur en `Literal` ; `GT`, `LOOK1/2`,
    `PARTITION` retirés. Inspiré de `ecolopharm-sales/…/core/repository/query_filters.py`.
11. ✅ **Ni variable globale ni fonction d'une ligne** (constantes dans leur classe, `max`/`min`/`//` de
    Python à la place de `lmax`/`lmin`/`floor_div`, génériques PEP 695).
12. 📝 **Schéma de sortie construit à partir de l'objectif** : la solution sort dans un schéma organisé
    comme une réponse de solveur de tournées (`summary`, `unassigned[]`, une entrée par ressource avec ses
    `steps[]`, `status`, `message`), construit dynamiquement : chaque terme minimisé s'agrège par son nom
    (p. ex. `setup`), les non-assignés se listent, et un type de sortie donné est rempli à partir de l'état
    ou de la solution.
13. 📝 **Infaisabilité rendue par le solveur**, tout de suite ou à la fin, selon **un seul booléen**.
14. 📝 **Peut-être un `AbstractSolverModel`** plus tard ; pas d'`AbstractSolver`.
15. ✅ **Constructeur = configuration, méthodes = données** (section 6) ; un argument propre à une méthode
    y reste.
16. ✅ **Générique** : `PartSpacing` retiré du catalogue (règle du domaine), vocabulaire de cours retiré
    de `modeling/`.
17. ✅ **Cours et étudiants = essais seulement** : le modèle de cours est sous `tests/`.
18. ❌ → ✅ en 23 **Pré-vérification : deux niveaux, 15 s au plus.** Les deux niveaux seuls sont faits (niveau 3
    retiré) ; les 15 s ne sont pas tenues (section 4) : laissé pour une autre session, sans ajuster les
    essais pour masquer l'écart (le cas lent est marqué `xfail`).

19. ⚠️ **Pas de constructeur vide** : d'abord fait avec `KernelSource.write` et `SchemaReader.load`
    en méthodes de classe, mais elles créaient encore un objet par un constructeur vide ; repris en 22.

20. ✅ **Deux revues `@dev` indépendantes** évaluent la conception et le respect des règles de la
    section 6, ou proposent une meilleure option justifiée ; les deux réviseurs et l'auteur doivent
    s'accorder. Ce qui se discute est d'abord écrit ici.

21. ✅ **Nécessité de chaque morceau** : les revues vérifient que chaque module, classe et méthode sert
    (règle 14), en s'inspirant du code de F-A.

22. ✅ **La méthode de classe est une règle, pas une question** : une classe sans argument au
    constructeur est bizarre ; sinon, mieux. Décidé avec les deux revues (6 bis, 2
    et 3) : `SchemaReader(root)` et `KernelSource(templates, n_elements, n_sets)` créé par la méthode de
    classe `write` ; plus aucun constructeur vide.

23. ✅ **Niveau 3 générique, 20 s fermes, liaison plus rapide** : deux agents (section 4.1) ;
    mélange de déterministe et de recherche locale permis ; 20 s au plus par exécution, toujours.
    **Non générique pour n'importe quel problème = échec.** Fait : trois niveaux, 20 s tenues
    (7,5 s à 29 448 blocs), liaison 7 à 8 fois plus rapide ; six jours toujours non prouvés (section 4).

24. ✅ **Remue-méninges des deux agents ensemble, puis retour au travail** (section 4.1).

25. ✅ **« Six jours consécutifs », est-ce seulement un essai ? Et un problème sans temps ?** Oui :
    c'est un essai tiré du modèle de cours (sous `tests/`). Il représente une famille de cas : une seule
    règle sur un groupe, impossible seulement quand tout le groupe est considéré. Le moteur ne connaît
    pas les jours. **Écart trouvé :** la pré-vérification exige aujourd'hui une décision entière avec
    horizon (`ValueError` sinon). Un problème d'affectation seule, sans temps, est donc refusé. À
    corriger : la décision entière et l'affectation deviennent chacune facultatives, au moins une
    requise, avec un essai sur un modèle sans temps.

26. ✅ **Pré-vérifier entité par entité, comme mes contrôles, mais génériques** (section 4.2) : pas
    d'optimisation ni de recherche pour prouver ; des bornes de charge tirées du modèle.

27. ✅ **Faire exactement la même chose que mes contrôles** : parité vérifiée par script, entité par
    entité, chiffres compris (section 4.2).

28. ✅ **Ne rien retirer avant d'avoir vérifié que ce code n'aide pas à reproduire mes contrôles de
    faisabilité** (section 4.2).

29. ✅ **Un arrêt de vérification se reprend là où il était** : quand F-A arrête les réviseurs, l'auteur
    les relance depuis leur point d'arrêt (2026-10-27, réviseurs A et B relancés).

30. ⏳ **Un agent Fable cherche du temps** (liaison, niveaux 1 à 3) par de petites optimisations, sans
    complexifier le code ; une simplification à temps égal ou meilleur est encore mieux. Il mesure
    aussi le parallélisme sur tous les cœurs moins deux (jamais tous). Rappel : `ForkPool` (1 contre
    8 cœurs) n'avait rien gagné quand la liaison dominait, et la liaison ne domine plus. L'agent
    travaille sur une copie et livre un correctif mesuré, appliqué après les revues en cours.
    **Objectif précisé par F-A : plus simple et plus rapide.** Environ 7 s, ce n'est pas mal. Beaucoup de
    fichiers ou d'opérations complexes pour une fraction de seconde : pas la peine. Une simplification,
    ou quelques secondes gagnées simplement (le parallélisme, par exemple) : ça vaut la peine.

31. ✅ **Règles ajoutées** (section 6, règles 15 à 21, et compléments aux règles 7 et 9) : ce qui avait
    été discuté sans être écrit. Valider à la construction (`model_validator`), pas dans une propriété
    comme `bound_mask`. Du code qui n'apporte rien est de la sur-ingénierie : `UnsupportedError` ne
    porte rien et sera remplacée par une exception standard.

32. ⏳ **Chemins de schéma reconnus par l'IDE** (remplacé par la section 1 ter : F-A refuse les clés et les
    déclarations d'ensembles et de paramètres). `m.schema.tasks.duration` passe par `__getattr__` :
    IntelliJ ne le voit pas. À faire : des sélecteurs typés en lambda, par élément, par exemple
    `tasks = m.set(lambda s: s.tasks, key=lambda t: t.id_)` puis
    `m.param(lambda task: task.duration, over=tasks)`. Le modèle devient générique sur son type de
    schéma (`Model[S]`, `IndexSet[T]`), et l'IDE vérifie chaque attribut. À l'exécution, la lambda
    reçoit un enregistreur qui note le chemin, comme `SchemaPath` aujourd'hui. La règle 6 gagne une
    exception : une lambda qui sélectionne un chemin est une déclaration, pas une aide d'une ligne.

33. ⏳ **Pas de `level` : une contrainte est dure, `minimize` est souple et porte un poids.**
    `constraint(expr, name=)` est toujours dure ; `minimize(expr, name=, weight=)` est toujours souple.
    Le poids vient de la configuration de la définition, plus de `bind(data, weights)`. Les primitives
    (`no_overlap`, `cumulative`, `bin_packing`) sont dures. `ConstraintLevel` disparaît de l'API du
    modèle ; le noyau garde ses deux axes, dur avant souple.

34. ⏳ **Schéma de sortie, séparé du modèle** (suite de 12). Le modèle décrit l'entrée et les règles ;
    la sortie est une autre déclaration. Elle associe la solution (début et cible par élément, termes
    minimisés par nom) à un type Pydantic de sortie, avec les mêmes sélecteurs typés. L'entrée et la
    sortie ne se mélangent pas.

35. ✅ **`Reference` dans `scripts/compare_course_definition.py`** : ce sont les contraintes écrites à la
    main de F-A, reconstruites pour comparer leurs comptes avec les 13 règles du modèle d'essai. Ce
    n'est qu'un outil de comparaison, hors du moteur.

36. ✅ **Revue du 2026-10-27 (réviseurs A3 et B3) : la pré-vérification donnait encore de fausses
    preuves**, jamais sur les données réelles mais sur d'autres modèles (règle 9) :
    - pauses (`Breaks`) : une période partielle, un bloc à cheval sur deux périodes ;
    - règles par ensemble lues sur des ensembles que le groupe ne touche pas ;
    - éléments de durée nulle.

    En plus :
    - des explications incomplètes (la pause, les filtres du niveau 1) ;
    - `bin_packing` mal borné dans un modèle avec temps ;
    - `where=` ignoré par les pauses ;
    - un groupe de locaux signalé deux fois ;
    - un script de parité qui ne compare pas ce que rend `run(bound)`.

    Une passe de corrections est en cours ; chaque correction vient avec un essai écrit d'abord. Sur
    les données réelles, CP-SAT confirme chaque surcharge vérifiée (6/6 à 600 enseignants, 5/5 à 4 900).
    **À trancher par F-A** : l'état d'une exécution (`deadline`, `kernel`, domaines) reste-t-il sur
    l'objet après `run(bound)` (6 bis n° 8) ou devient-il local (règle 20) ?

37. ✅ (décidé, à faire) **Plus de `via` ni d'indice imbriqué : le sélecteur suit la structure des
    données.** Les données restent telles qu'elles arrivent, imbriquées : pas de copie à plat. Une
    référence se suit dans la lambda, par la clé de l'ensemble visé :
    ```python
    recipes = m.set(lambda s: s.recipes, key="id_")
    tasks = m.set(lambda s: s.tasks)
    rooms = m.set(lambda s: s.resources, key="id_")
    processing_time = m.param(
        lambda task: recipes[task.recipe_id].scheduling_options[0].processing_time, tasks
    )
    options = m.subsets(lambda task: recipes[task.recipe_id].scheduling_options, tasks, of=rooms)
    start = m.var("start", tasks, size=processing_time, high=5 * 24)
    room = m.assign("room", tasks, rooms, among=options)
    ```
    `size=` est le nom générique (celui de CP-SAT) de ce qu'un élément occupe ; `m.var` n'a plus de
    `length=`. Dans un sélecteur, `recipes[...]` est une recherche par clé, et la liaison en fait la même
    jointure rapide qu'aujourd'hui. La clé est une chaîne : l'IDE ne la vérifie pas, mais un nom qui
    n'est pas un champ du type est refusé dès la définition du modèle, avec la liste des champs.

38. ⏳ **La pré-vérification tourne sur le modèle lié, et rien d'une exécution ne reste sur un objet**
    (règle 20). `ModelCompiler.bind(data)` rend un modèle lié : le noyau et ce que la liaison a lu
    (gabarits, données). `ModelCompiler` ne garde plus « les gabarits de la dernière liaison ».
    `InfeasibilityPrecheck(config).run(bound)` rend les constats, et tout le reste est local. Le délai
    de 20 s couvre alors la pré-vérification seule, la liaison étant faite avant.

39. ✅ **Pas de `gc.freeze()`** : ≈ 0,06 s à la taille réelle (1 336 blocs), pas de quoi toucher tout le
    processus.

40. ✅ (décidé, à faire) **La pré-vérification n'a pas de configuration.** Les seuils (0,9 et 0,5) et les
    budgets deviennent des constantes de la classe. `time_limit` devient un argument :
    `InfeasibilityPrecheck.run(bound, time_limit=20)`, une méthode de classe, sans instance (règle 2).

41. ✅ (décidé, à faire) **Reconnaissance par l'IDE : une lambda par champ lu** (option A). Les chemins
    `m.schema.…` disparaissent :
    ```python
    tasks = m.set(lambda s: s.tasks)
    recipes = m.set(lambda s: s.recipes, key=lambda r: r.id_)
    recipe = m.param(lambda task: task.recipe_id, tasks, of=recipes)
    length = m.param(lambda r: r.scheduling_options[0].processing_time, recipes)
    start = m.var("start", tasks, length=length[recipe], high=5 * 24)
    ```
    `Model[S]`, `IndexSet[T]` et `Parameter` sont génériques : chaque lambda reçoit un objet typé
    (`CourseProblemInstance`, `CourseBlock`, `Recipe`) et l'IDE vérifie chaque attribut. À la liaison,
    la lambda est appelée sur un enregistreur qui note le chemin. La lecture reste donc vectorisée comme
    aujourd'hui, et la lambda ne sert qu'à nommer le chemin. Une lambda qui ne fait que choisir un
    chemin est une déclaration, pas une aide d'une ligne : la règle 6 l'admet. **Pas de lambda qui va
    chercher un objet dans un dictionnaire pour en lire un champ** (la proposition refusée de la
    section 1 ter).

42. ✅ (décidé, à faire) **Noms alignés sur MiniZinc, pauses à part, listes.**
    - `any_of` → `exists`, `all_of` → `forall`, `count_over` → `count`, `sum_over` → `sum`
      (`min_over` → `min`). CP Optimizer et CP-SAT n'ont ni `each` ni `count_over`.
    - `m.within(lambda task: task.time_windows, block)` : un membre d'une liste lue sur un élément. C'est
      le générateur `w in windows[t]` de MiniZinc, et on lit ses champs directement (`window.start`).
    - `m.breaks(teams, start, name=, period=, window=[11, 14], length=)` : une primitive à part, sur les
      mêmes groupes que la non-chevauchement, comme les pauses de véhicules d'OR-Tools. Plus de classe
      `Breaks` ni d'objet dans `no_overlap`. `Transition` reste dans `no_overlap`, car elle change la
      règle elle-même.
    - Des listes plutôt que des tuples dans l'API (`window=[11, 14]`).
    - `m.var(..., span=processing_time, ...)` : `span=` remplace `length=`.

43. ✅ **Passe de corrections terminée (2026-10-27).**
    - Essais : 74 réussis et 1 échec attendu ; ruff propre ; mypy propre hors fichiers orphelins.
    - Fuzzers des réviseurs : ≈ 1 500 modèles, compilé et Python pur. 0 fausse preuve, 0 explication
      fausse. Les seules erreurs sont les refus voulus d'un horizon qui n'est pas fait de périodes
      entières.
    - Corrigé :
      - les pauses (horizon en périodes entières, sinon refus) ;
      - les éléments de durée nulle ;
      - les règles par ensemble ;
      - `bin_packing` avec temps ;
      - les explications ;
      - un groupe de locaux signalé une seule fois ;
      - des entités nommées ;
      - `UnsupportedError` remplacée par `ValueError` ;
      - `bound_mask` retiré.
    - Temps : 1,0 s à 3 656 blocs ; 10,7 s à 29 448 blocs, sur une machine à charge 2,6.
    - Le correctif Fable n'est pas appliqué tel quel : il vise un fichier depuis réécrit. Ses deux idées
      passent à la ronde de l'API.
    - Le backend de F-A a changé le 27 (configuration métier passée aux contrôles, fichier
      `CONSOLIDE.xlsx` modifié et refusé par l'analyseur) : `compare_checks.py` est adapté et tourne
      sur le fichier du 26.
    - La comparaison ne compte désormais que les entités notées d'un côté ou de l'autre :
      - enseignants : 3 notés, mêmes demandes 3/3, mêmes offres 1/3 ;
      - groupes de locaux : 31 notés, même verdict 30, mêmes demandes 28, mêmes offres 9.

      Les écarts sont à expliquer après la ronde de l'API, car les contrôles de F-A ont aussi changé.

44. ✅ **Ronde de l'API** (37, 38, 40, 41, 42, 33), confiée à un nouvel agent sur une copie de départ
    (`scratchpad/before_api/`), avec les mêmes garanties : comptes identiques, 0 fausse preuve, règles
    1 à 21.

45. ✅ **Un seul fichier maître fait foi : `data/gabarit_CMV_FICHIER_MAITRE_v13_CONSOLIDE.xlsx`** (celui
    que F-A dépose sur Google Drive) ; les autres versions de `data/` sont ignorées. Son en-tête H3 de
    `5_GROUPES_COURS` est bien `regroupement_id` (129 identifiants). L'analyseur le refuse encore pour
    une seule valeur : la ligne 380 (Hébert, 03582, groupe 351-3T0-MV_001) porte `3T0_A4_001` au lieu
    de `3T0_MV_001`. C'est le point Q8 du suivi, pas encore appliqué. Le code n'a pas de défaut.

46. ✅ **Ronde de l'API terminée (2026-10-27), vérifiée par l'auteur.**
    - Essais : 80 réussis et 1 échec attendu ; ruff propre ; mypy propre hors fichiers orphelins.
    - `modeling/` n'importe plus rien de `cegep_mv_schedule`.
    - Équivalence avec l'arbre d'avant : mêmes comptes, variations et constats, compilé et Python pur.
      Cas couverts : cours 60 et 600, ateliers, flottes, livraisons, 20 cas de ressources. Le texte du
      noyau est identique pour les modèles inchangés.
    - Fuzz sur les mêmes graines : 0 fausse preuve, 0 explication fausse.
    - Temps à 29 448 blocs : liaison ≈ 2 s, pré-vérification 5,6 à 6,3 s.
    - Correction en passant : `block_day_spacing` était un `minimize` dur. Ce sont maintenant deux
      `constraint`, avec les mêmes comptes, et ses violations apparaissent enfin dans `violations`.

    Points ouverts, à trancher par F-A :
    1. la forme des équipes : `group_by(lambda t: t.teacher_id, tasks, extra=…, skip=…)` ;
    2. `min`/`max` à plusieurs valeurs renommés `smallest`/`largest`, parce que `min` est désormais
       l'agrégat ;
    3. `subsets` écrit avec une compréhension `[o.resource_id for o in …]` ;
    4. l'état d'une exécution dans un enregistrement local gelé `_Run` ; un `set_absent` fait avant
       `run(bound)` n'est pas gardé.

    `CONSOLIDE.xlsx` est encore refusé (ligne 380), donc pas de comparaison sur données réelles.

47. ✅ **Ronde suivante (décidée par F-A le 2026-10-27)** :
    - **La liaison reste à l'intérieur.** Le seul point d'entrée est `compiled.bind(data)`, qui rend le
      modèle lié. Le service ne voit ni `Binder` ni ce qu'il a lu. `Binder(model)` est créé par
      `ModelCompiler` ; `expand(data)` rend ce qui a été lu, que le modèle lié garde sous un nom
      interne, sans attribut public `binder`. Plus de `expand` sans argument (règles 1 et 3).
    - **`group_by` à une seule clé, sans `extra=` ni `skip=`.** La clé peut être une liste : l'élément
      entre dans chaque groupe de sa liste, et dans aucun si elle est vide. La règle « qui enseigne ce
      bloc » vit dans les données : une propriété `teacher_ids` sur une copie d'essai du bloc, sous
      `tests/`. Rien dans le code de F-A.
    - **`run(bound)` rend le noyau exactement comme il l'a reçu** : placement et éléments absents.
    - `m.largest(0, a, b)` et `m.smallest(...)` gardent leurs arguments positionnels.

48. ✅ **Règle 22 : pas de classe `_Privée`** ; une représentation intermédiaire reste dans le fichier qui
    s'en sert, nommée normalement.

49. ✅ **Ronde 47 vérifiée par l'auteur** (l'agent a été arrêté pendant une campagne de fuzz compilé
    inutile).
    - Ce qui est fait :
      - `Binder(model).expand(data)` rend un `Reading`, et `BoundModel` est dans `compiled_model.py` ;
      - plus aucune classe `_Privée` ;
      - `group_by(lambda t: t.teacher_ids, tasks)`, avec `teacher_ids` sur une copie d'essai du bloc ;
      - `run(bound)` rend le noyau tel qu'il l'a reçu (un essai le vérifie).
    - Vérifié :
      - essais : 81 réussis et 1 échec attendu ; ruff et mypy propres ;
      - équivalence avec la copie d'avant : 36 cas identiques sur 36, compilé et Python pur ;
      - fuzz en Python pur : identique, 0 fausse preuve.

50. ✅ **Vérifier à la mesure du changement** (F-A) : « est-ce que ça marche ou pas ». Les essais et une
    comparaison avant/après suffisent. Pas de campagne lourde, comme compiler un noyau par modèle
    aléatoire pendant des heures.

51. ✅ **Le moteur suit les changements du backend (2026-10-28)**, vérifié sur le seul fichier maître
    du jour (point 45).
    - `scripts/compare_checks.py` reprend le nouveau `TeacherLoadCheck` : `_binding_departments`, les
      heures `open_by_day` des blocs assemblés, la note de `_teacher_issue` (`TeacherFeasibilityCheck`
      est retiré du backend), et `IndexedSheets` reçoit l'assemblage d'essai comme dans `run_loads`.
    - Écarts restants, tous des avertissements : 04909 est `OVERLOADED` chez F-A et prouvé au niveau 2
      ici (ses 6 blocs de 202-001-RE et 202-SN1-RE) ; 06187 et 10391 n'ont que des blocs réservés, que
      F-A ignore et que la pré-vérification compte comme une offre égale à la demande (`NEARLY_FULL`) ;
      07788 et 10255, même note, par des blocs co-enseignés.
    - **Le parallélisme revient** (point 30, règle 21) : sur le fichier du jour, 7 recherches sur 592
      font tout le temps (3 indécises au budget plein, 1 groupe prouvé et son explication). Le passage
      témoin reste en ordre ; les groupes indécis sont cherchés par `ForkPool` dans `os.cpu_count() - 2`
      copies ; l'omission d'un groupe couvert par un groupe prouvé s'applique ensuite, dans l'ordre.
      15 à 19 s → 10 à 13 s, constats identiques (section 5). Ce qui reste est surtout l'explication
      du groupe prouvé, en ordre par nature. Sans groupe indécis, rien n'est forké.
    - **Une charge épinglée n'est plus « presque pleine »** : quand chaque élément n'a qu'un placement
      (un début, une cible) et que l'offre égale la demande, la charge n'est que les données ; seule
      une surcharge (une preuve) reste. Sur le fichier du jour, cela retire CH-147, 06187, 07788, 10255
      et 10391, que F-A ignore aussi. Un camion à 9/10 dont le colis n'a que lui reste averti.
    - **Un groupe non décidé est nommé** (`FindingKind.UNDECIDED`, jamais prouvé) au lieu de
      disparaître : fichier du jour, 3 groupes (360-223-RE, 101-C2D-MV/101-SN2-RE, 351-1T1).
    - Écart nommé (règle 17) : les preuves sont identiques partout ; seuls ces 5 avertissements
      partent et ces groupes indécis arrivent (plus un en Python pur, 60 enseignants, germe 2, où la
      limite de 20 s coupe la recherche).

52. ✅ **Revue des signatures (2026-10-28, F-A) : règle 2 relue, règle 8 levée pour trois propriétés,
    enregistrements à la place des tuples.**
    - **Règle 2 relue.** Elle dit qu'une classe sans rien à configurer ne s'instancie pas, non que la
      pré-vérification n'a pas d'instance. Le point 40 est renversé : les limites (`GROUP_LIMIT`,
      budgets, ratios, `WORKERS`, délai) sont un `PrecheckConfig`, et
      `InfeasibilityPrecheck(config).run(bound)` est une méthode d'instance, comme la règle 1 l'écrivait
      déjà. Le `run` reste un
      enregistrement passé aux aides (règle 20).
    - **Règle 8 levée pour trois propriétés** : `CourseBlock` (backend) porte `teacher_ids`,
      `split_keys` et `spacing_keys`, les clés de regroupement que le modèle de cours lisait sur un
      sous-type d'essai. `KeyedCourseBlock`, `KeyedCourseBlocks` et `KeyedCourseInstance` sont
      supprimés : `CourseModelDefinition.schema = CourseProblemInstance`, et l'alias `CourseModel`
      part avec eux (`Model[CourseProblemInstance]` se lit tel quel). `split_keys` suit la règle du
      backend (`_build_split_groups`) : l'identifiant de base avant `_part<n>`.
    - **Le noyau lit et rend des enregistrements** (règles 23 et 24) : `search(plan, stop)
      -> SearchOutcome`, `repair(plan, stop)`, `lone_placements(element, candidates,
      filters) -> LoneDomain`, avec `SearchPlan`, `Candidates`, `Filters` dans `kernel.py`. Plus de
      `blame` passé en sortie : chaque résultat porte le sien. Dans la pré-vérification, `_plan` rend le
      `SearchPlan`, `_search` un `SearchOutcome`, `_group_proof` un `GroupProof`, `_domain` un
      `LoneDomain`. `stop` reste un argument : il est propre à l'appel, pas à la liaison.
    - **Le modèle de cours, sans `CourseSymbols`** : comme les autres modèles d'essai, `define` déclare
      en locales et chaque règle reçoit ce qu'elle lit (au plus `model` et trois symboles) ; `bloc`,
      `spacing` et les tables sont déclarés dans les règles d'espacement, `is_lab` et `course` dans les
      règles de labo.
    - **Le moteur place et relit lui-même**, pour tout modèle : `BoundModel.place(placements)` et
      `BoundModel.solution`. Plus d'encodeur-décodeur (règle 8). Les ex æquo d'une liste triée se
      départagent par clé (`rank`, posé à la liaison).
    - **`ModelSolution`** (Pydantic gelé, sans `arbitrary_types_allowed`) : `placements` (élément,
      valeur, fin, cible : des objets du schéma), `unassigned`, `summary`. Le `Summary` sort de
      `BoundModel.solution` sans rien mapper : un `TermSummary` (valeur, poids, coût) par
      `minimize(name=)`, les violations par `constraint(name=)`. Pas d'`instance` : l'appelant a déjà
      le problème, et chaque `Placement` porte déjà ses objets.
    - **Pas de mapper : un schéma de sortie** (`modeling/solver/output.py`, F-A, 2026-10-28). `Output`
      est un modèle Pydantic dont chaque champ dit où il lit dans la solution, par `validation_alias` et
      `AliasPath` ; ce qui se dérive des champs est un `computed_field`. `Schedule[Out]` hérite de
      `ModelSolution` et ajoute `solution: Out`, la seule part validée : un horaire rendu se redonne
      tel quel comme départ à chaud. Le schéma reste un pur mapping ; il n'hérite pas du moteur.
    - **Le service et le solveur sont génériques** : `OptimizationService(compiled, solver,
      output).solve(problem, initial)` fait `compiled.bind`, `solver.solve`, `Schedule.of(solution,
      output)` ; `KernelSolver(config)` n'hérite de rien. Le seul code propre aux cours hors de la
      définition est le schéma de sortie, `tests/course_schedule.py`.
    - `scripts/course_flow.py` écrit le flux en ligne, chaque bloc sous le nom de sa classe.
    - Vérifié (2026-10-29) : 89 essais, 0 différence sur les 13 règles (fichier v13, 20 placements),
      et `prove_course_output.py` identique sur les quatre placements : local et créneaux de chaque
      bloc, objectif, compte de chaque règle, réservations tenues.
    - **Règles 23 et 24 dans tout le moteur** : plus aucun tuple rendu ni reçu. `kernel.placement`,
      `variables`, `group.pair`, les `children` et les colonnes de `shape` sont des listes ; une
      paire de types différents est un enregistrement (`Through`, `GeneratedKernel`, `Nested`,
      `Ranging`, `ComputedRead`, `Cells`). Les aides de `codegen.py` rendent des lignes sans
      retrait : l'appelant indente, `pad` n'est plus passé. `ElementIndex` perd
      `arbitrary_types_allowed`.
    - Écarts gardés, avec leur raison : les tuples clés de dictionnaire ou d'ensemble (il faut un
      hachable) et ceux du tas (`heapq` compare des tuples) ; les fonctions Cython de la boucle
      chaude et `Kernel.__init__` (même exemption que `compiled_constraints.py` du backend) ; les
      constructeurs d'enregistrement, qui reçoivent leurs champs (règle 1) ; l'API publique
      `breaks`, `no_overlap`, `var`, `cumulative` et `minimize`, à options nommées depuis que le
      point 42 a retiré l'objet `Breaks`.
    - **Recherche d'un placement d'un groupe** (`solver/infeasibility/placement_search.py`), sur le
      modèle de l'`Acceptor` du backend : `PlacementSearch(stop)` dit ce qui est lu une fois un
      élément placé (`reads`), ce qui est lu en avant (`ahead`), et cherche (`search`).
      `ExhaustiveSearch` essaie chaque candidat, profondeur d'abord, en élaguant : une impasse prouve
      qu'il n'y a pas de placement. `HeuristicSearch` cherche un placement, profondeur d'abord puis
      conflits minimaux : une impasse ne prouve rien. Elles remplacent le drapeau `witness` et ses
      trois branches ; une autre recherche s'ajoute sans toucher la pré-vérification. Le
      `SearchOutcome` du noyau porte `verdict` et `blamed` : plus de second enregistrement. Mêmes
      constats, même temps, mesuré contre le premier commit sous la même charge.
    - **L'arrêt est le `StopCondition` du backend** (F-A, 2026-10-29) : une recherche reçoit
      `AnyOf([MaxIterations(plafond), *run.stops])`, où une itération est une évaluation du noyau.
      `run.stops` est vide sans délai réglé, `[MaxTime(time_budget_seconds)]` sinon : pas de 20 s
      par défaut. Pour ne pas demander `should_stop` à chacune des 20 millions d'évaluations
      possibles, `StopCondition` gagne `checkpoint(iter_idx)`, la prochaine itération où demander
      (`MaxIterations` : son plafond, `MaxTime` : 4 096 plus loin, `NoImprovement` : là où la
      patience s'épuise, `AnyOf`/`AllOf` : le plus tôt des enfants). Par défaut, l'itération
      suivante : `SearchLoop` ne l'appelle pas et ne change pas. `_explain` reçoit `run.stops`.
      Vérifié : 91 essais, `prove_course_output.py` identique ; sur v13, sans délai, mêmes constats
      (309/310 enseignants, 56/58 bassins) en 13 s : chaque recherche atteint son plafond avant.
    - **Repris de `core/optimization`** (F-A, 2026-10-30) : `Summary.score` est un `Score` du backend
      (dur d'abord, `is_feasible`) au lieu de deux entiers ; le nombre de copies de la
      pré-vérification vient d'`available_cores()`, qui respecte les cœurs permis à un conteneur, là
      où `os.cpu_count()` compte ceux de l'hôte ; le germe de `repair`, écrit 12345 dans le noyau,
      devient `PrecheckConfig.seed` (même défaut), passé par `HeuristicSearch(stop, seed)`.
      `SolverConfig` ne sert pas ici : il porte un délai de 30 s par défaut, et la pré-vérification
      n'en a pas. Non repris : `FreezePolicy` (une réservation fige déjà le bloc par ses fenêtres,
      à l'assemblage ; le backend lui-même ne l'appelle que dans ses essais), et la boucle, les
      acceptations, `FeasibleReached` et `NoImprovement` tant que `KernelSolver` ne cherche pas.
      Vérifié : 91 essais, sortie identique, mêmes constats sur v13.

53. ✅ **La recherche du backend portée sur le noyau (F-A, 2026-10-30)**, pour comparer les deux.
    - **Repris tel quel de `core/optimization`** : `SearchLoop`, `HillClimbingAcceptor`,
      `RouletteWheel`, `AnyOf(MaxIterations, NoImprovement, MaxTime)`, les contrats `Evaluator`,
      `SearchState`, `AbstractMove`, `MoveSelector`.
    - **Depuis le 2026-11-02, copiés dans le moteur, sans les abstractions** : la boucle dans
      `KernelSolver`, la roulette dans `DestroySelector`, `Score` et les arrêts dans `solver/`. Plus
      aucune classe abstraite : un seul état, un seul coup, un seul évaluateur. Vérifié sur v13 : mêmes
      placements aux graines 1, 2 et 3, mêmes 36 constats ; puis le tirage d'un « opérateur de
      réparation » unique, sans effet, est retiré, ce qui change la suite des tirages.
    - **Porté de `course_assignment/alns/`, sans rien de propre aux cours** :
      - `CourseAlnsEvaluator` devient `KernelEvaluator` : chaque règle comptée autour des éléments
        déplacés, avant et après. Plus de jumeaux : le noyau généré tient déjà les règles.
      - `CourseDestroyRepair` devient `DestroyRepair` : même ordre (famille la plus serrée, puis
        le plus long), paire co-début en un balayage, retour des éléments sans place, alignement
        des paires scindées tant que `full_hard % 1000`.
      - `CourseAlnsSelector` devient `DestroySelector` : les mêmes 9 bras et plafonds. « Enseignant »
        devient « famille », « local » devient « ensemble », « jour » devient « période ».
      - `CourseAlnsSolver` devient `KernelSolver.solve` : construction gloutonne sans départ,
        `soft_mode`, la boucle, puis le retrait des conflits durs (le plus dur, à égalité celui qui
        laisse le moins de mou).
    - **Ce qui bouge ensemble est déclaré par le modèle, là où la règle est posée** :
      `model.tie("no_teacher_overlap", Tie.RELEASED)` (les familles),
      `model.tie("co_start", Tie.PAIRED)`, `model.tie("split_block_spacing", Tie.GROUPED)`, et la
      période sur la décision, `model.var(..., period=slots_per_day)`. `SearchIndex.of(bound)` les
      lit ; la configuration du solveur ne nomme aucune règle. Les liens viennent de la
      pré-vérification (`InfeasibilityPrecheck.read`), comme les domaines : les débuts et les
      ensembles qu'aucune règle dure sur l'élément seul n'interdit (les fenêtres et les locaux
      permis). Sur v13 : 310 familles, 115 paires, 1 groupe, comme le backend ; les 94 blocs sans
      enseignant ne sont dans aucune famille, comme dans le backend.
    - **Noyau** : `relocate(element, partner, domains, draw)`, `element_hard`, `violating`,
      `around_set_count` et `set_placement` publics. Un élément absent est d'abord remis où il
      était : les écarts se lisent entre placements présents. Vérifié sur 300 déplacements de v13 :
      l'écart rendu égale toujours celui d'un recompte complet. `bind` prend `lex_base=100_000`
      par défaut, la base du backend.
    - `prove_course_output.py` place et relit sans chercher : il compare un même placement.
    - **Résultat** (`compare_search.py`, v13, 2 000 itérations, patience 400, les deux placements
      notés par le moteur) :

      | Graine | Backend placés / mou / s | Moteur placés / mou / s |
      |---|---|---|
      | 42 | 1 282 / 5 460 / 16,1 | 1 286 / 5 270 / 13,6 |
      | 1 | 1 282 / 5 450 / 14,3 | 1 284 / 5 790 / 14,3 |
      | 2 | 1 284 / 5 440 / 11,7 | 1 286 / 5 280 / 13,1 |
      | 3 | 1 281 / 5 670 / 12,8 | 1 279 / 5 510 / 14,6 |

      Dur à 0 partout. Le moteur place autant de blocs, au bruit des graines près, avec un mou du
      même ordre, dans le même temps. Vérifié : 99 essais, sortie identique, mêmes constats sur v13.

54. ✅ **Banc d'extraction sur le fichier maître réel (F-A, 2026-11-03)**, pour mesurer l'extraction
    de contraintes (`docs/dominance breaking/pipeline.py`) sur de vraies données, avec le moteur
    seulement : « générique, sinon c'est un échec ». Le paquet `mining/` et ses scripts, livrés dans
    `extraction_contraintes_v8.zip`, sont installés ici et portés sur `ModelCompiler`.
    - `scripts/real_bench.py` : quinze bancs, chacun une poignée de départements du fichier maître
      (`MasterInstance`) et des familles de règles cachées au modèle connu (`RuleFamily` :
      indisponibilités, réunions, fenêtres de cours, espacement, battement, groupes de locaux,
      fermetures). Le planificateur simulé est `KernelSolver` sur le modèle de cours avec ses propres
      poids et peu de patience (`PlanDrawer`) ; les références sont une marche sur les règles connues
      à partir d'un placement du moteur (`FeasibleWalk`, `start=`) et les placements du moteur avec
      son objectif. `CourseTables` écrit les tables, `CourseTruth` les règles cachées dans les mots des
      tables, `Grader` donne à chaque règle son sort (règle, question, manquée, invisible) et à chaque
      ligne rapportée son étiquette (vraie, connue, effet, fausse). Les plans qui laissent plus de
      `max_absent` blocs sans place sont écartés : le fichier maître en laisse un à trois par
      département, de toute façon.
    - `mining/` : `Plan.absent` (les éléments laissés de côté, que la table ne montre pas),
      `FeasibleWalkConfig.start` (une marche qui part d'un plan quand un départ au hasard n'atteint
      pas les règles), `KernelRestarts` (les plans de la recherche du moteur, une graine chacun),
      `span_lengths` (les durées lues sur le modèle lié, partagées par la recherche et les tables).
    - Le pipeline d'extraction ne change pas : il reçoit des tables et un `config.json`, rien d'autre.
      Résultats et constats dans `docs/dominance breaking/evaluation_reelle.md`.

55. ✅ **Départ à chaud et valeurs fixées, génériques (F-A, 2026-11-03)** : « le départ à chaud, et
    fixer une variable dans la solution », pour tout modèle, pas seulement le cégep.
    - `VariableAssignment(element, value, target)` : les valeurs des variables d'un élément, par les
      clés que déclarent ses ensembles (sa position, en texte, sans `key=`). Une part `None` reste
      libre. Le même objet sert aux deux entrées : `KernelSolver.solve(bound, warm_start=, fixed=)`.
    - `warm_start` : où la recherche commence ; ce que le modèle ne sait pas lire est sauté.
      `fixed` : ce que la recherche ne change pas ; refusé en entier si le modèle ne sait pas le
      lire, ou si les règles dures de l'élément seul l'interdisent.
    - Une valeur fixée **coupe le domaine** de l'élément (`SearchIndex.of(bound, fixed)`), et celui
      de sa paire `PAIRED`. Les mouvements ne tirent que dans les domaines : aucun opérateur ne teste
      « fixé ». Un élément fixé n'est jamais laissé de côté ; un élément tenu en place (un seul
      départ, un seul ensemble) n'est jamais relâché par une destruction.
    - `ModelSolution` ne garde que des clés (`assignments`, `unassigned`, `summary`) : elle survit au
      JSON et se redonne telle quelle. Les objets du problème que lit le schéma de sortie sont
      résolus dans `Schedule.of(bound, output)`, en `AssignedElement` (lu sous `assigned`) ;
      `Placement` disparaît.
    - Clés vérifiées une fois, à la liaison : chaînes, sans doublon.
    - La recherche ne demande plus `period=` : sans période, tout l'horizon en est une. Elle tourne
      donc aussi sur l'atelier (`job_shop_model`) et sur le chargement de camions sans temps
      (`bin_packing_model`), essayés avec des valeurs fixées et un départ à chaud.
    - Limite : un conflit entre deux éléments fixés n'est pas refusé avant la recherche (retirer les
      éléments libres pour le tester fausserait les règles d'agrégat) ; il reste compté dans
      `summary`.
    - Essais : 122 réussis ; ruff et mypy propres.

56. ✅ **Revue indépendante et refactorisation (F-A, 2026-11-03)** : trois réviseurs `@dev`
    (noyau et recherche ; écriture et liaison ; architecture), puis les corrections.
    - **Défauts corrigés (règle 16)** :
      - `SchemaReader` indexait un ensemble par son seul chemin : deux ensembles sur la même liste,
        l'un `distinct`, l'autre non, ou l'un avec `key`, se remplaçaient sans erreur selon l'ordre de
        lecture. L'index est maintenant par (chemin, clé, `distinct`) ; les clés d'un regroupement par
        famille, non par nom.
      - `ModelSolution` ne survivait pas au JSON en Python pur (entiers NumPy) : le résumé est en
        entiers Python.
      - Une clé entière (`key="num"`) refusait toute liaison : les clés voyagent maintenant en texte
        (`str`), doublons toujours refusés. Remplace « clés : chaînes » de la ronde 55.
      - `Kernel.remove` vérifie l'élément, comme `insert`.
      - `DestroyRepair.PROTECTED_WEIGHT = 1000`, un poids du backend sans sens ici (toute règle dure
        pèse 1) : l'alignement des paires tourne tant qu'un conflit dur reste.
    - **Pré-vérification dans le service** : `OptimizationService(compiled, precheck, solver, mapper)`,
      puis `solve(problem, warm_start, fixed, *, check)`. Une seule liaison sert la pré-vérification et
      le solveur ; `OptimizationResult.precheck_issues` vaut `None` quand `check=False`. `PrecheckConfig.up_to` (un
      `PrecheckLevel` : `ELEMENT`, `GROUP`, `COMPONENT`) dit jusqu'où aller.
    - **Le compilateur n'entre pas au constructeur du solveur** : lier est l'étape coûteuse, et un
      `BoundModel` ne naît que de `bind`. Son type garantit qu'il est lié ; aucune vérification de plus.
    - **Organisation** : `compiled_model.py` → `model_compiler.py` (fait par F-A) ; les
      enregistrements d'une résolution et `output.py` → `solution.py` ; `op.py` fondu dans
      `expression.py` ; `modeling/` → `optimization/` (fait par F-A). `KernelSolver.solve` rend sa
      solution, reprise telle quelle par le service.
    - **Une seule façon (règle 4)** : `ModelCompiler.names`, `ModelCompiler.variables` et
      `Model.is_hard` retirés ; `Model.hard`, `Model.unit_weights` et `Model.variables` servent la
      liaison et l'évaluateur. `InfeasibilityPrecheck.read(bound)` est une méthode de classe : la
      recherche ne crée plus une pré-vérification dont elle ignore la configuration (règle 2).
      `KernelSource.write` reçoit les rangs, plus de tableau provisoire.
    - **Noms** : `Kernel.search` → `depth_first`, `Kernel.repair` → `min_conflicts`, `_take` →
      `_counts_around`, `_change` → `_change_since`, `_full_for` → `_count_full`, `olds` →
      `before` ; `DestroyRepair.remove` → `remove_first` ; `Template.free` → `has_free_members` ;
      `TermShape.bound` → `aggregated_symbols` ; `Binder.elements` → `decided` ;
      `BoundModel.unreadable` → `unreadable_reason` ; `BoundModel._templates` → `templates`.
      `KernelSource` garde son nom (F-A).
    - **`Finding` → `PrecheckIssue`** (F-A : « ce n'est pas des constats ») : `FindingKind` → `IssueKind`,
      `finding.py` → `precheck_issue.py`, `findings` → `precheck_issues`. Une entrée
      est une preuve, un avertissement de charge ou une recherche restée ouverte : un « problème »
      signalé par la pré-vérification, pas toujours un constat prouvé.
    - **Une forme d'affectation, une sortie configurée** (F-A : « deux structures pour les mêmes
      affectations » ; « passer une configuration, et tout se fait »). `AssignedElement` et
      `Schedule` sont retirés ; `VariableAssignment` est la seule forme d'une affectation. Le schéma
      de sortie reste la configuration du problème (un `BaseModel` par alias, décision du 2026-10-28 : pas de
      mapper écrit à la main) ; `OutputMapper(schema).map(bound, solution)` le remplit. Les entrées
      sous `assigned` sont des dictionnaires (un chemin d'alias parcourt des dictionnaires ou des
      objets, pas les deux) : élément et cible y sont lus par `model_dump`. Aucune fin d'intervalle
      dans `ModelSolution` : le moteur la calcule à la sortie, par le `span` du modèle.
      `OptimizationService(compiled, precheck, solver, mapper).solve(...)` rend un
      `OptimizationResult(solution, output, precheck_issues)`. Un dépôt (MongoDB ou autre) viendrait
      au constructeur, sans que `solve` nomme le problème. Vérifié : même `CourseSolution`, champ par
      champ, sur 1 262 blocs du fichier v13 ; `prove_course_output` identique au backend.
    - **`Reading` → `BoundData`** (F-A : le nom ne dit pas ce que c'est) : les données du problème
      telles que le modèle les lit (éléments, cibles, valeurs, sous-ensembles en positions, et les
      énoncés liés). `BoundModel.reading` → `BoundModel.data` ; le paramètre `data` de
      `ModelCompiler.bind` et de `Binder.expand` devient `problem`.
    - **`PrecheckRun` se lit et répond sur ses données** (F-A : un enregistrement rempli au
      constructeur par une autre classe n'est pas une exécution). `solver/infeasibility/precheck_run.py`
      : `PrecheckRun.read(bound, stops)` lit ce que les règles dures lient puis le domaine seul de
      chaque élément ; `domain`, `array`, `ready`, `linked_groups(limit)` et `components` sont ses
      méthodes. `PrecheckRun.isolated(kernel)` remplace `_kept` et les `set_absent` épars : tout
      élément absent dans le bloc, le noyau rendu tel quel après. `stops` reste dans l'exécution
      (règle 22 : l'état d'une exécution est son enregistrement ; le sortir mettrait deux méthodes à
      cinq arguments, règle 23). `HardStructure` → `HardTies`. `SearchIndex` lit `PrecheckRun.read`
      sans passer par la classe de preuve. `precheck.py` passe de 1 600 à 1 050 lignes, `precheck_run.py` en a 555.
    - **`InfeasibilityPrecheck` → `InfeasibilityCheck`** (F-A).
    - **Règle 17** : constats, comptes et solutions identiques à la mesure du début de la ronde, compilé
      et Python pur, cours, atelier et camions.
    - **Vitesse** : `ElementIndex` en `dataclass` ; `_per_element` coupe le chemin une fois.
    - **Règle 17** : comptes à la liaison, solution d'une recherche à graine fixe, constats de la
      pré-vérification, identiques avant et après, compilé et Python pur, sur les cours (fichier v13),
      un atelier et un chargement de camions. Seul le texte généré change (les noms du noyau).
    - **Proposé, non fait** : un tableau de colonnes par nœud à la place des curseurs de `codegen`
      (`next_const`, `next_collection`) ; une seule convention d'indentation dans `codegen` ;
      sortir la lecture de structure de `precheck.py` (1 600 lignes) ; fondre `KernelState` dans le
      noyau ; compiler `kernel.py` une fois et le `cimport` (≈ 90 % des 14 s de compilation).
    - Essais : 132 réussis ; ruff et mypy propres (sauf `output.py`, à supprimer).

57. ✅ **`mining/` relu et réécrit, `scripts/` trié (F-A, 2026-11-03)** : le paquet venait d'un zip,
    installé tel quel (règle 26). Deux réviseurs `@dev` indépendants, puis les corrections.
    - **Une seule recherche** : `mining/local_search.py` (une deuxième recherche à côté de
      `KernelSolver`) et `OptimizedRestarts` retirés ; les plans de référence viennent de
      `KernelRestarts` ou de `FeasibleWalk`, qui marche depuis le placement du noyau dans les domaines
      de `SearchIndex` et abandonne après `max_thinnings` tours.
    - **Un seul évaluateur** : `Candidate.post` pose un candidat comme règle dure, nommée d'après lui,
      sur les seuls éléments de sa clé (`where` sur une table de drapeaux). `ConstrainLoop` juge les
      plans dans un modèle portant tous les candidats ; `broken`, `calendar` et `durations` retirés.
      `CandidateEncoder` (une `ABC`, une seule implémentation) et `BenchEncoder` remplacés par
      `ConstrainedDefinition` (règle 5).
    - **`Findings`** : le `findings.json` de l'extraction, typé, hors de `Candidate`.
    - **Banc à règles cachées** dans `tests/mining/planted_bench.py` : `PlantedBench`, `BenchTables`,
      `BenchTruth` ; règles en enregistrements typés ; les superpositions fusionnées dans le vrai
      problème (heures bloquées héritées, enseignants et cours fusionnés exclus des tirages) ; moins de
      13 enseignants refusé.
    - **`solver/binding/`** : `model_compiler`, `binder`, `bound_collection`, `schema_reader`.
    - **Scripts** : `jobshop_precheck_check` et `example_course_model` retirés ; `absent_kernel_check`
      et `generic_groups_check` devenus des essais ; les classes de `compare_checks`,
      `prove_course_output`, `compare_search`, `compare_course_definition` et `scale_precheck` dans
      `tests/parity/` et `tests/large_course_problem.py`, celles de `real_bench` dans
      `tests/mining/real_bench.py` (règle 27). `scale_precheck` affiche les groupes et composantes
      cherchés, sans délai par défaut (`--budget` signale les passes coupées).
    - Deuxième revue (deux réviseurs) : superpositions (lots d'espacement hérités), rôles et clés
      inconnus refusés, règle « au plus n par jour » d'une cible comptée par cible, banc essayé
      (`planted_bench_test.py`). Un calcul avec les règles plantées : 1,0 s au lieu de 0,8 s sans.
      Bancs : `objective_tradeoff_check` 4 min 29 → 1 min 41 ; mêmes résultats (28°).
    - Règles 28 (code qui respire) et 29 (les essais suivent l'arborescence) ajoutées.
    - Essais : 170 réussis ; ruff propre ; mypy propre (sauf `output.py`, à supprimer).

### Fichiers à supprimer à la main

La suppression de fichiers est refusée à l'agent ; rien ne les importe :

```bash
cd docs/fixes
git rm -f modeling/interpreter.py modeling/explanation.py modeling/collector.py \
  modeling/template.py modeling/errors.py \
  modeling/problem_formulation/global_constraint.py \
  modeling/solver/element_classes.py tests/cpsat_capacity_test.py \
  tests/definitions/__init__.py
rm -rf tests/definitions
```

Et depuis la ronde 47 : `git rm -f modeling/solver/bound_model.py` (`BoundModel` vit maintenant dans
`compiled_model.py`).

Et depuis la ronde 54 : `rm "../dominance breaking/real_bench.py"` (un premier banc sur le solveur du
backend, remplacé par `scripts/real_bench.py`).

Et depuis la ronde 55 : `rm modeling/solver/decision.py` (un premier nom de `VariableAssignment`,
jamais importé).

Et depuis la ronde 56 (rien ne les importe ; `output.py` gêne mypy tant qu'il reste) :

```bash
cd docs/fixes
git rm -f optimization/solver/output.py optimization/problem_formulation/op.py \
  optimization/solver/infeasibility/finding.py
rm scripts/jobshop_precheck_check.py scripts/example_course_model.py \
  scripts/absent_kernel_check.py scripts/generic_groups_check.py \
  tests/solver/binding/generic_schema_test.py
```
