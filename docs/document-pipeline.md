# Extraire d'un document ce qu'un schéma demande

Ce module lit un document et en extrait, section par section, ce que le projet décrit dans un
**schéma d'extraction** (une sous-classe de `SectionEntities`). Le projet fournit le schéma et
quelques réglages ; lire, découper, trier et extraire est fait pour lui. Il ne s'agit pas seulement
de lire un PDF : c'est une démarche en quatre étapes, dont chacune existe parce qu'une version plus
simple a échoué sur un vrai corpus :

1. **lire** une source (un PDF, un classeur) en un arbre de mise en page ;
2. **indexer** ce document en sections : du code détecte les titres candidats, un modèle décide
   lesquels sont de vraies sections ;
3. **classer** les sections : trancher la pertinence quand le titre ne suffit pas, recoller ce que
   la détection a coupé en deux ;
4. **extraire** ce que chaque section nomme, par couches : le modèle juge, le code décide.

Tout vit dans `core/llm/document_extraction/` : `document_parser.py`, `layout/` et
`sheet_parser.py` (étape 1), puis un service par étape, `indexing/` (`IndexingService`),
`classification/` (`ClassificationService`) et `extraction/` (`ExtractionService`), et les
enregistrements que les étapes s'échangent (`section.py`, `indexed_document.py`,
`section_entities.py`). `orchestrator/document_extraction_orchestrator.py` montre comment s'en
servir : un orchestrateur qui enchaîne les étapes pour un fichier.

## L'exemple qui court

Une entreprise exploite des machines sur plusieurs sites. Son archive d'entretien mêle :

- des **rapports d'inspection** en PDF, de quelques pages à plusieurs centaines, chacun en
  sections numérotées (« 4.2 Compresseur C-12 — essais de pression ») ;
- des **procès-verbaux du comité technique**, où chaque point approuve ou reporte une
  intervention ;
- des **bons de travail** numérisés, sans couche de texte ;
- un **classeur** d'inventaire dont chaque feuille décrit un site.

Le schéma d'extraction demande, pour chaque section pertinente : les constats, un résumé, le
coût, et les codes qu'elle nomme (bons de travail `BT-2026-0145`, machines `C-12`), en
distinguant ceux qu'elle porte de ceux qu'elle ne fait que citer.

## Les étiquettes de coût

Chaque étape dit ce qu'elle coûte :

| Étiquette | Ce que l'étape fait |
|---|---|
| **code** | Du Python seul : déterministe, reproductible, gratuit. |
| **modèle, texte** | Un appel de modèle sur du texte : des lignes, une section, des cartes. |
| **modèle, vision** | Un appel de modèle sur des images : des pages rendues. |

Le principe qui traverse tout le module : **le modèle n'est appelé que là où une règle de code ne
peut pas trancher**, et sa réponse est ensuite bornée par du code.

## 1. Lire une source — `core/llm/document_extraction/`

`DocumentParser(config).parse(contenu)` lit un fichier : un classeur passe par `SheetParser`, un
PDF par `LayoutParser` (pypdfium2, via cffi pour la vitesse). Un projet ne règle que ce qui varie
d'un corpus à l'autre, dans `LayoutConfig` : l'en-tête et le pied de page répétés, et la détection
des deux colonnes, par exemple `DocumentParser(LayoutConfig(columns=ColumnConfig(enabled=False)))` ;
`SheetParser.parse(contenu)` lit un classeur Office (openpyxl). Les deux rendent le même
`LayoutDocument` figé : des pages, des blocs, des lignes, des empans (taille, police, gras,
italique, souligné), les images (raster et dessins vectoriels distingués), les liens sortants,
l'outline embarqué, la somme SHA-256 des octets. Le champ
`origin` dit d'où vient l'arbre : `"points"` pour une page, `"cells"` pour une feuille.

Ce que le parseur règle une fois pour toutes, pour que personne en aval ne le refasse :

- **l'en-tête et le pied de page** (`FurnitureMarker`) : une ligne qui se répète au même endroit
  sur une fenêtre de pages est du mobilier. Chaque page reçoit le verdict, et tout ce que les
  étapes lisent passe par `body_lines`, `body_images`, `body_links` ou `content_band` ;
- **l'ordre de lecture** (`ColumnReorder`) : une page sur deux colonnes (un texte bilingue côte à
  côte) se lit colonne gauche puis colonne droite, chaque ligne pleine largeur gardant sa place.
  Un tableau, ou un formulaire dont la colonne d'étiquettes est étroite, n'est jamais réordonné ;
- **le texte hors page** : un caractère peint hors de la boîte de la page n'est pas lu.

Seul `PageRenderer` rouvre les octets, pour rendre en PNG les bandes de page que l'extraction
demande.

### Brancher une autre source

Une étape ne lit jamais un fichier : elle lit un `LayoutDocument`. Brancher une source (un `.docx`,
un courriel, une page web), c'est écrire un parseur qui rend cet arbre :

- une **page** (`origin="points"`) a une géométrie en points : l'indexation lit sa typographie,
  l'extraction peut rendre ses pixels ;
- une **feuille** (`origin="cells"`) n'a pas de pixels : une feuille est une page, chaque ligne de
  cellules une ligne de texte, le nom de la feuille une entrée d'outline. L'indexation la juge
  contre son propre corps de texte, l'extraction la lit en texte seulement.

Le parseur se branche dans `DocumentPipelineOrchestrator.parse`, à côté des deux existants : c'est
le seul endroit où le format est lu.

## 2. Indexer — `core/llm/document_extraction/indexing/`

`IndexingService.index(document)` rend la liste des `LocatedSection` du document : détecter,
réconcilier, localiser.

### Détecter (code)

`HeadingFinder` (page) ou `SheetHeadingFinder` (feuille), choisi sur `origin`, relève des
**candidats** de deux sources : l'outline embarqué et la typographie (une ligne plus grande, plus
grasse ou soulignée que le corps du document, mesuré sur ce document). La détection vise le
**rappel** : elle ne tranche rien, elle s'assure que toute vraie section est parmi les candidats.

Ce que les mesures ont appris :

- **Un outline non corroboré n'est jamais montré au modèle.** Les outlines sont souvent périmés
  ou tronqués par l'exportateur. Une entrée d'outline fusionne avec un candidat typographique
  seulement si leurs boîtes se chevauchent **et** si leurs textes concordent ; sinon elle est
  retirée. L'outline confirme, il ne crée jamais.
- **Le profil du corps se mesure par document** : la taille modale et la norme de gras d'un
  rapport ne sont pas celles d'un autre.

### Réconcilier (modèle, texte)

`SectionReconcilerAgent` (configuré par `SectionReconcilerConfig`) décide quels candidats sont de
vraies sections et rend, pour chacune, son
titre, son ouvreur (`section_number`), sa page, son **niveau** et sa **pertinence**.

- **Le niveau est une décision du modèle, jamais une extraction.** Quatre itérations de règles
  (jeton de tête, profondeur des points) ont échoué : un même motif est un numéro d'ordre dans un
  document et un code de référence dans un autre. Le niveau se lit sur l'énumérateur quand il
  existe ; là où rien n'est numéroté, une rubrique crée de la profondeur. Le code ne parse jamais
  ces champs.
- **L'ouvreur est conservé, mais ce n'est pas une identité.** Il sert à replacer la section sur sa
  ligne imprimée (`printed_heading` le recolle au titre), jamais à la nommer.
- **La pertinence se juge sur ce que la détection montre**, pas sur le corps que personne n'a
  encore lu. `uncertain` est une réponse valide : deux titres interchangeables peuvent cacher des
  sujets opposés. Le classement est asymétrique par construction : une fausse alerte coûte une
  lecture, un faux rejet perd une section en silence. Dans le doute, jamais `relevant`. Le critère
  de pertinence est une configuration du projet (`relevance_criterion` : « concerne une
  intervention sur une machine »), écrite dans le prompt ; le reste du prompt est générique.
- **Un long document se lit en fenêtres de pages, l'une après l'autre.** Un appel unique tient
  jusqu'à quelques centaines de pages ; au-delà, `PageWindow` découpe les candidats en fenêtres
  de `window_pages` pages. Chaque fenêtre reçoit de la précédente un `WindowCarry` : les conventions réglées et
  la chaîne des rubriques encore ouvertes à la coupure, pour qu'une rubrique imprimée avant la
  coupure donne leur niveau aux items qui suivent. Le recollage est déterministe : une fenêtre ne
  garde que les sections qui commencent sur ses pages. Un document plus court qu'une fenêtre reçoit
  exactement le prompt d'avant la découpe.

### Localiser (code)

`HeadingLocator` replace chaque section sur ses lignes imprimées (score flou sur
`printed_heading`), lui donne le texte qui court jusqu'à la section suivante, et récolte les
**références de fichiers** : les annotations de lien de la bande de la section qui pointent vers
un format lisible (`FileReference`). Une référence ne va jamais dans un prompt.

Enfin, `SerialItemSplitter` sépare un item d'une série numérotée (« 6.3 », « 6.4 ») que le
modèle a fondu dans sa voisine, seulement quand une ligne s'ouvre sur l'item promu ; un code cité
au milieu d'une phrase ne coupe jamais.

## 3. Classer — `core/llm/document_extraction/classification/`

Ce n'est pas une étape de la chaîne, c'est un paquet de **juges** appelés là où leur question se
pose.

- **La pertinence** (`ClassificationService.settle`, qui interroge `SectionRelevanceAgent` :
  modèle, texte), tout de suite après l'indexation, sur les seules
  sections `uncertain` : une fenêtre glisse sur le texte que le localisateur vient de découper
  jusqu'à ce qu'une tranche décide. Si aucune ne tranche, la section reste `uncertain` et sera lue
  quand même. Il passe avant l'extraction pour qu'une section jugée hors sujet ne coûte aucune
  lecture.
- **La réparation des sections** (`ClassificationService.repair` : code puis modèle, texte), après
  l'extraction, parce que ses indices sont les entités extraites. Le détecteur
  (`MergeCandidateDetector`, avec les genres d'identité que le projet déclare) juge chaque frontière entre deux sections
  voisines : même mention d'identité, section sans identité sous une section qui en porte une, ou
  au moins deux signaux faibles communs. `SectionRepairAgent` tranche `merge` ou `separate` sur la
  fin de la première et le début de la seconde. Une fusion joint les textes, unit les références et fusionne
  les entités ; une suite de fragments se replie en chaîne.

## 4. Extraire — `core/llm/document_extraction/extraction/`

Le schéma d'extraction appartient au projet : il hérite de `SectionEntities` et y ajoute ses
champs. Les descriptions de ses champs **sont** les instructions du modèle. Le module ne lit que
deux champs de la base : les `mentions` (ce que la section nomme par une valeur imprimée, avec son
genre et son rôle, `own` ou `cited`) et `dated`. Tout le reste voyage jusqu'aux modèles d'aval sous
forme de cartes.

Une seule lecture qui répondait à tout s'est trompée sur près d'une valeur sur deux, surtout en
typant mal un code imprimé ou en recopiant un code des exemples de ses propres instructions.
L'extraction lit donc **par couches**, chacune ne faisant que ce qu'elle fait bien :

1. **La moisson** (code) : `MentionHarvester`, configuré par des `MentionFormat` (un genre, une
   expression régulière), trouve chaque valeur candidate imprimée dans la section.
2. **Le texte, toujours** (modèle, texte) : `SectionEntitiesAgent` lit la section ; les candidats lui sont listés
   dans le message utilisateur (le prompt système reste un préfixe statique que le serveur met en
   cache) et il dit lesquels la section porte ou cite. Sa réponse est ensuite **bornée à la
   moisson** par du code : une valeur inventée n'y survit pas. Pour un genre sans format, une
   valeur que le texte n'imprime pas est retirée.
3. **La vision, au besoin** (modèle, vision) : `SectionVisionAgent`, seulement sur les pages dont les pixels portent ce
   que le texte ne porte pas (page numérisée, images couvrant une bonne part de la page), rendues,
   réduites et envoyées par lots sous un budget de jetons. Une page n'est jamais coupée entre deux
   sections (`VisionRouter`). Les valeurs lues en vision sont bornées à la moisson quand la section a une couche de
   texte.
4. **La fusion** (code) : `SectionEntities.merged_with` ; le texte gagne tout recouvrement, la
   vision n'ajoute que sous une clé neuve.
5. **Rien ne se perd** : une section que le modèle ne sait pas lire est gardée, vide, avec la raison
   dans `extraction_error`. Une feuille se lit en texte, avec le même schéma.

`ExtractionService[Out]` est générique sur le schéma du projet : `extract(sections, document,
contenu)` rend les sections dans le même ordre, chacune avec ses entités.

Le projet d'origine ajoutait des **passes étroites**, chacune déclenchée par un test de code bon
marché (une question unique avec citation exigée sur une section qui imprime un mot de décision ;
une lecture des cartouches de plans recadrés et rendus en grand). Elles sont propres à son domaine
et ne sont pas reprises ; le mécanisme se recopie : un lecteur de plus, derrière un test sans
modèle, dont la réponse est vérifiée contre le texte.

## Enchaîner — `orchestrator/document_extraction_orchestrator.py`

`DocumentExtractionOrchestrator(parser, indexing, classification, extraction)` reçoit le
`DocumentParser` configuré par le projet (classeur ou PDF) et les trois services ; il ne fait que
les enchaîner. `run(fichier)` traite **un** fichier, sans rien savoir des autres :
lire la source, indexer, trancher les sections `uncertain`, extraire, réparer les sections, et
rendre un `IndexedDocument`. Le format est lu là, et nulle part après ; le document est parsé une
fois et toutes les étapes lisent le même arbre. Chaque fichier se traite seul, donc en parallèle
des autres.

## Ce qu'un projet fournit

| Quoi | Où | Exemple |
|---|---|---|
| Le schéma d'extraction | une sous-classe de `SectionEntities` | `findings`, `summary`, `cost` |
| Les genres de mentions et leurs formats | `SectionEntitiesConfig.kinds`, `MentionFormat` | `work_order: BT-\d{4}-\d{4}` |
| Le critère de pertinence | `relevance_criterion` du réconciliateur et du juge de pertinence | « concerne une intervention sur une machine » |
| Les genres qui identifient une section | `SectionRepairConfig.identity_kinds` | `work_order`, `approval` |
| Les modèles | les constructeurs des agents | un modèle par agent, choisi par mesure |

## Ce qui n'est pas repris

Ce qui est propre au domaine du projet d'origine reste chez lui : le contexte d'un document (quelle
instance l'a produit, à quelle réunion), les profils de formats de codes par émetteur, les réparations de
preuve propres à ses textes, le consensus entre plusieurs lectures, l'export vers un entrepôt, et la liaison des sections
de plusieurs documents en un graphe puis leur assemblage en dossiers. Une
idée mérite d'être retenue : **le modèle le plus fort rédige les formats de codes d'un émetteur,
le code les vérifie contre tout le corpus (ils compilent, ils sont imprimés au moins deux fois, ils
ne chevauchent aucun autre genre), une personne approuve** ; seuls les formats approuvés bornent
l'extraction.
