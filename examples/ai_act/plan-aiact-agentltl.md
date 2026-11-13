# AI Act × AgentLTL — Plan d’analyse et d’action

**Rulebooks d’agents qui font respecter, appel d’outil par appel d’outil, les obligations du règlement (UE) 2024/1689 (AI Act)**

*Version du 12 novembre 2026 — état du droit intégrant l’Omnibus numérique sur l’IA (règlement (UE) 2026/1744, en vigueur depuis le 27 juillet 2026).*

> **À lire d’abord.** Aucun rulebook ne peut à lui seul *garantir* la conformité à l’AI Act : une grande partie du règlement porte sur l’organisation (qualité, documentation, gouvernance, formation) et sur le *fond* des décisions (la classification est-elle juste ? la revue humaine est-elle réelle ?). Ce que AgentLTL garantit, c’est autre chose, et c’est précieux : **qu’un agent ne peut pas exécuter une action qui court-circuite une étape de conformité**, et que chaque refus est tracé. Les rulebooks ci-dessous transforment les obligations de l’AI Act en *préconditions procédurales vérifiées avant chaque appel d’outil*. Ce document n’est pas un avis juridique ; la qualification de vos systèmes doit être validée par votre juriste ou DPO.

---

## Table des matières

1. Résumé exécutif
2. Analyse de l’AI Act au 12 novembre 2026
3. Analyse d’AgentLTL
4. Méthode : traduire une obligation en règle vérifiable
5. Architecture cible
6. Matrice de couverture article → règle
7. Les rulebooks (AGENTLTL.yaml)
8. Variante Python pour les agents métier
9. Tests et validation
10. Plan d’action phasé
11. Gouvernance, preuves et indicateurs
12. Limites et risques résiduels
13. Sources

---

## 1. Résumé exécutif

**Le constat.** Les instructions données à un agent (prompt système, CLAUDE.md, politique interne) ne sont que des conseils : l’agent peut les perdre après une longue conversation, dans un sous-agent ou sous la pression de terminer. Pour des obligations légales assorties d’amendes allant jusqu’à 35 M€ ou 7 % du chiffre d’affaires mondial, un conseil ne suffit pas.

**La réponse.** AgentLTL contrôle chaque appel d’outil d’un agent *avant son exécution* à partir d’un fichier `AGENTLTL.yaml`, avec une logique temporelle (ordre des appels, valeurs liées entre appels, mémoire entre sessions). Il fonctionne avec Claude Code, GitHub Copilot CLI, Mistral Vibe, Codex, et comme bibliothèque Python pour n’importe quelle boucle d’agent.

**Le livrable.** Huit modules de règles (44 règles AI Act, plus des règles de la bibliothèque AgentLTL), un serveur d’outils de conformité à construire (`aiact`), une suite de tests de non-régression, et un plan de déploiement en cinq phases.

**L’urgence.** Le **2 décembre 2026, dans 20 jours**, deux obligations deviennent applicables : la nouvelle interdiction des systèmes générant des contenus intimes non consentis ou pédopornographiques (art. 5), et la fin du délai de grâce du marquage lisible par machine (art. 50(2)) pour les systèmes génératifs mis sur le marché avant le 2 août 2026. Les modules 00, 01 et 02 sont à déployer en priorité.

---

## 2. Analyse de l’AI Act au 12 novembre 2026

### 2.1 Calendrier applicable

| Date | Ce qui s’applique | Statut au 12/11/2026 |
|---|---|---|
| 2 février 2025 | Pratiques interdites (art. 5), maîtrise de l’IA (art. 4) | **En vigueur** |
| 2 août 2025 | Modèles d’IA à usage général — GPAI (art. 53 à 55), gouvernance | **En vigueur** |
| 27 juillet 2026 | Omnibus numérique sur l’IA (règlement (UE) 2026/1744) | **En vigueur** |
| 2 août 2026 | Transparence (art. 50), régime de sanctions, pouvoirs de sanction de la Commission sur les GPAI | **En vigueur** |
| **2 décembre 2026** | Nouvelle interdiction art. 5 (contenus intimes non consentis, CSAM) ; fin du délai de grâce art. 50(2) pour les systèmes génératifs déjà sur le marché | **Dans 20 jours** |
| 2 août 2027 | Bacs à sable réglementaires nationaux | À venir |
| 2 décembre 2027 | Haut risque « autonome » — annexe III (emploi, crédit, éducation, biométrie, infrastructures critiques, services publics…) | À venir (reporté du 2 août 2026) |
| 2 août 2028 | Haut risque intégré à des produits réglementés — annexe I | À venir (reporté du 2 août 2027) |
| 2 août 2030 | Systèmes à haut risque existants utilisés par des autorités publiques | Inchangé |

L’Omnibus **décale les dates** du haut risque ; il ne supprime ni n’allège ce qui sera exigé. Les éléments de preuve (journaux, revues humaines, documentation) se constituent dans la durée : un agent qui respecte les règles dès aujourd’hui produit l’historique qu’un auditeur demandera en décembre 2027.

### 2.2 Les rôles qui déterminent les obligations

| Rôle | Qui | Obligations clés pour un agent |
|---|---|---|
| **Fournisseur** (art. 16) | Développe le système ou le met sur le marché sous son nom | Gestion des risques, données, documentation technique, journalisation, contrôle humain *par conception*, évaluations, marquage art. 50(2) |
| **Déployeur** (art. 26) | Utilise le système sous son autorité | Contrôle humain effectif, conservation des journaux (≥ 6 mois), information des personnes, FRIA (art. 27) pour certains déployeurs, divulgation art. 50(1)/(4) |
| **Fournisseur de GPAI** (art. 53-55) | Développe un modèle à usage général | Documentation, politique droit d’auteur (opt-out TDM), résumé des données d’entraînement ; + évaluations adversariales et signalement d’incidents si risque systémique |

**Point d’attention agents :** un déployeur qui utilise un agent pour une finalité à haut risque non prévue par le fournisseur peut devenir lui-même fournisseur (art. 25). Le module 03 doit alors être activé côté déployeur.

### 2.3 Obligations retenues pour les rulebooks

Le critère de sélection : l’obligation se manifeste-t-elle dans **une action de l’agent** (un appel d’outil) que l’on peut autoriser, refuser ou conditionner ?

| Famille | Articles | Ce que l’agent fait concrètement | Contrôlable par AgentLTL ? |
|---|---|---|---|
| Interdictions | 5(1)(c)(d)(e)(f)(g)(h), 5 (Omnibus) | Appeler un outil de notation sociale, de moissonnage facial, de reconnaissance d’émotions au travail, de génération d’images | **Oui** (interdiction directe ou contrôle préalable) |
| Interdictions « sémantiques » | 5(1)(a)(b) manipulation, exploitation des vulnérabilités | Rédiger un message manipulateur | **Non** — relève du modèle et de classifieurs de contenu |
| Transparence | 50(1)(2)(3)(4) | Envoyer un message, publier une image, un texte, une vidéo | **Oui** (précondition par destinataire / par contenu) |
| Haut risque | 6, 9, 10, 12, 14, 26, 27, 86 | Appliquer une décision RH/crédit, déployer, changer de modèle, entraîner | **Oui** (préconditions + validation humaine) |
| GPAI | 53, 55 | Collecter des données, publier un modèle | **Oui** |
| Incidents | 73 | Détecter un incident, revenir en arrière sur un modèle | **Oui** (délai et gel des preuves) |
| Organisationnel | 4, 17, 18, 43, 49 | Former, certifier, enregistrer dans la base UE | **Non** — hors périmètre, mais tracé dans le plan d’action |

---

## 3. Analyse d’AgentLTL

### 3.1 Ce que fait l’outil

AgentLTL est un garde placé entre l’agent et ses outils. À chaque appel proposé, il évalue les règles du fichier `AGENTLTL.yaml` sur la **trace** des appels déjà exécutés, et refuse l’appel s’il viole une règle, *avant* qu’il ne s’exécute, en expliquant pourquoi et comment se conformer.

Quatre capacités le distinguent d’un simple filtre de commandes, et ce sont précisément celles dont l’AI Act a besoin :

| Capacité AgentLTL | Ce qu’elle permet pour l’AI Act |
|---|---|
| **Ordre des appels** (`before`, `since`, `finally`, LTL passée : `Y`, `O`, `H`, `S`) | « Pas de décision sans revue humaine préalable », « pas de publication sans marquage » |
| **Variables liant les appels** (`$r`, `$a`, `$c`) | La divulgation doit concerner *ce* destinataire ; le marquage *ce* contenu ; la revue *ce* dossier |
| **Mémoire inter-sessions** (`scope: project`) | Une FRIA faite il y a trois semaines compte encore ; une divulgation faite hier aussi |
| **Analyse réelle du shell** | `helm upgrade … -n prod` est reconnu quelle que soit la position de `-n` |

### 3.2 Les grammaires disponibles

| Forme | Sens | Usage AI Act |
|---|---|---|
| `never: T` | Aucun appel correspondant à T | Pratiques interdites |
| `before: {first: A, then: B}` | B exige un A antérieur | Préconditions (divulgation, marquage, revue, FRIA) |
| `before: {…, since: S}` | A doit suivre le dernier S | Documentation/évaluations à jour depuis la dernière modification |
| `require: T` | Les arguments de T doivent correspondre | Étiquette « généré par IA » obligatoire |
| `at_most: {call: T, times: n}` | Plafond | Lutte contre le biais d’automatisation (art. 14(4)(b)) |
| `finally: T` | Avant de rendre la main, T a eu lieu | Journalisation de clôture, signalement d’incident |
| `ltl: '…'` | Formule libre | Gel des preuves après incident (art. 73(6)) |

### 3.3 Les modes de sanction

| Mode | Effet | Usage dans les rulebooks |
|---|---|---|
| `stop` | Refus, et plus aucun outil ne s’exécute avant réponse de l’humain | Pratiques interdites, intégrité des journaux et des règles |
| `block` | Refus systématique, non contournable par l’agent | Préconditions de conformité |
| `ask` | L’humain reçoit une demande d’autorisation motivée | Contrôle humain art. 14, cas limites (satire, données sensibles) |
| `retry` / `warn` | Refus avec possibilité d’insister | **Non utilisés** pour l’AI Act (l’agent ne doit pas pouvoir passer outre) |
| `log` | Autorisé, mais signalé | Phase de rodage (« shadow mode ») |

### 3.4 Limites de l’outil à connaître

L’outil ne voit pas l’intérieur des scripts (`python script.py`, `make target`) ; les commandes non analysables (`eval`, `$CMD args`) relèvent du réglage `unparseable` ; un appel refusé ne compte jamais comme « fait » ; la mémoire de session repart de zéro à chaque session ; les variables ne se combinent pas encore avec `since` ; et du côté `first` d’une règle à variables, seules des valeurs `with` sont admises (pas `succeeded`). Les rulebooks en tiennent compte.

---

## 4. Méthode : traduire une obligation en règle vérifiable

### 4.1 Grille de traduction

| Type d’obligation | Question à se poser | Forme AgentLTL |
|---|---|---|
| Interdiction absolue | Quel outil, avec quels arguments, ne doit jamais être appelé ? | `never` + `stop` |
| Interdiction contextuelle | Le contexte est-il déclaré dans les arguments ? | `require` (contexte obligatoire) + `never` (contexte interdit) |
| Précondition | Quelle étape de conformité doit précéder l’action, *pour le même objet* ? | `before` + variable + `scope: project` |
| Fraîcheur | L’étape doit-elle être refaite après une modification ? | `before … since` |
| Clôture | Que faut-il avoir fait avant de rendre la main ? | `finally` |
| Délai | En combien d’actions au plus ? | `ltl: within_steps(...)` |
| Supervision | Où un humain doit-il trancher ? | `mode: ask` |

### 4.2 Trois principes de conception

**1. AgentLTL vérifie l’ordre ; le serveur d’outils vérifie le fond.** Une règle `before` prouve que `check_tdm_optout` a été appelé pour le domaine, pas que le domaine n’a pas exprimé d’opposition. L’outil de conformité doit donc *échouer* ou *refuser* lui-même quand le fond n’est pas bon, et l’outil métier (le crawler) doit consulter le même registre : défense en profondeur.

**2. Les arguments décisifs viennent du serveur, pas de l’agent.** Une règle sur `context: workplace` ne vaut que si l’agent ne peut pas mentir sur le contexte. Idéalement, `context`, `source` ou `category` sont injectés par la configuration de déploiement ou calculés par le serveur MCP, et l’agent ne les fournit pas librement.

**3. Un outil de validation humaine doit impliquer un humain.** `human_review`, `human_editorial_review` et les modes `ask` ne valent contrôle au sens de l’art. 14 que si l’outil attend la décision d’une personne identifiée et compétente (workflow d’approbation, signature), et ne retourne jamais une approbation que l’agent aurait produite lui-même.

### 4.3 Convention de nommage

Dans les harnais (Claude Code, Copilot CLI, Vibe, Codex), les outils MCP s’appellent `mcp__<serveur>__<outil>`. Les rulebooks utilisent :

- `mcp__aiact__<outil>` pour les **outils de conformité** (serveur à construire, voir §5) ;
- `"mcp__*__<outil>"` pour les **outils métier**, quel que soit le serveur qui les expose (CRM, RH, CMS, vision…) ;
- `aiact_<sous-commande>` pour la **CLI interne** `aiact`, utilisée par les agents de code (module 07).

Si vos outils portent d’autres noms, adaptez les cibles ; `agentltl validate` signale toute règle qui ne pourrait jamais se déclencher.

---

## 5. Architecture cible

```
            ┌──────────────────────────────┐
 Agent ───► │  AgentLTL (garde)            │  ← AGENTLTL.yaml (modules 00 à 07)
 (Claude    │  refuse / demande / laisse   │  ← mémoire session + projet
  Code,     └──────┬───────────────────────┘
  Codex,           │ appels autorisés seulement
  Python…)         ▼
     ┌─────────────────────────┐   ┌─────────────────────────────┐
     │ Serveur MCP « aiact »   │   │ Serveurs MCP métier         │
     │ classify, fria, review, │◄─►│ CRM, RH, CMS, vision, MLOps │
     │ disclose, mark, log…    │   │ (consultent le registre)    │
     └──────────┬──────────────┘   └─────────────────────────────┘
                ▼
     Registre de conformité (compliance/) + journaux d’audit en ajout seul
```

### 5.1 Spécification du serveur MCP `aiact`

| Outil | Arguments | Comportement attendu | Articles |
|---|---|---|---|
| `classify` | `system` | Enregistre la qualification (interdit / haut risque / transparence / minimal) validée par un humain dans `compliance/` | 6, annexe III |
| `fria` | `system` | Ouvre et suit l’analyse d’impact droits fondamentaux ; réussit seulement si validée | 27 |
| `risk_review` | `system` | Revue du registre des risques avant changement | 9 |
| `human_review` | `case_id` | **Bloquant** : attend la décision d’un humain habilité, enregistre son identité | 14, 26(2) |
| `inform_affected_person` | `person_id` | Envoie l’information prévue et l’accès à l’explication | 26(11), 86 |
| `log_event` | `case_id`, contenu | Écrit dans le journal d’audit en ajout seul, horodaté | 12, 19, 26(6) |
| `disclose_ai_interaction` | `recipient` | Envoie la mention « vous échangez avec une IA » | 50(1) |
| `mark_synthetic_content` | `asset_id` | Appose le marquage lisible par machine (métadonnées type C2PA + filigrane) | 50(2) |
| `content_safety_check` | `request_id` | Classifieur ; **échoue** si la demande vise un contenu interdit | 5 (Omnibus) |
| `human_editorial_review` | `doc_id` | Revue éditoriale humaine sous responsabilité éditoriale | 50(4) |
| `inform_exposed_persons` | `session_id` | Information des personnes exposées | 50(3) |
| `datasheet`, `bias_examination` | `dataset` | Fiche de données, examen des biais | 10 |
| `check_tdm_optout` | `domain` | Lit robots.txt / TDMRep / métadonnées ; **échoue** si opposition | 53(1)(c) |
| `training_summary`, `doccheck`, `run_adversarial_evals` | — | Produisent les pièces GPAI ; échouent si incomplètes | 53, 55 |
| `flag_serious_incident`, `report_serious_incident`, `report_incident` | `id` | Qualification et notification d’incident | 55(1)(c), 73 |

### 5.2 CLI `aiact` (agents de code)

Même périmètre, en ligne de commande : `aiact classify --system X`, `aiact doccheck`, `aiact evals`, `aiact datasheet --dataset D`, `aiact mark --asset A`, `aiact fria --system X`. Chaque sous-commande renvoie un code de sortie non nul en cas d’échec, ce qui permet aux règles d’exiger `succeeded: true`.

---

## 6. Matrice de couverture article → règle

Couverture : **Forte** = l’action non conforme est physiquement empêchée ; **Partielle** = la règle impose l’étape, la qualité de l’étape dépend de l’outil ou de l’humain ; **Hors** = organisationnel.

| Article | Obligation | Règle(s) | Mode | Couverture |
|---|---|---|---|---|
| 5(1)(c) | Notation sociale | `aiact-art5-1c-notation-sociale` | stop | Forte |
| 5(1)(d) | Prédiction pénale par profilage | `aiact-art5-1d-prediction-penale-profilage` | stop | Partielle (argument `basis`) |
| 5(1)(e) | Moissonnage facial non ciblé | `aiact-art5-1e-moissonnage-facial`, `aiact-dev-bibliotheques-biometriques` | stop / ask | Partielle (argument `source`) |
| 5(1)(f) | Émotions au travail / à l’école | `aiact-art5-1f-contexte-declare`, `aiact-art5-1f-emotions-travail-education` | block / stop | Partielle (argument `context`) |
| 5(1)(g) | Catégorisation biométrique sensible | `aiact-art5-1g-categorisation-biometrique` | stop | Partielle |
| 5(1)(h) | Identification biométrique temps réel | `aiact-art5-1h-identification-temps-reel` | stop | Forte |
| 5 (Omnibus) | Contenus intimes non consentis / CSAM | `aiact-art5-ncii-csam-controle-prealable` | stop | Partielle (qualité du classifieur) |
| 5(1)(a)(b) | Manipulation, exploitation des vulnérabilités | — | — | Hors (modèle, classifieurs) |
| 6, annexe III | Classification avant mise en service | `aiact-art6-classification-prealable`, `aiact-dev-classification-avant-prod` | block | Partielle |
| 9 | Gestion des risques | `aiact-art9-revue-risques-avant-changement` | block | Partielle |
| 10 | Données, biais | `aiact-art10-fiche-donnees`, `aiact-art10-examen-biais`, `aiact-dev-fiche-donnees-avant-entrainement` | block | Partielle |
| 4a (Omnibus) | Données sensibles pour corriger les biais | `aiact-art4a-donnees-sensibles`, `aiact-art4a-suppression-apres-usage` | ask / block | Forte |
| 11, annexe IV | Documentation technique à jour | `aiact-dev-doc-technique-a-jour`, `aiact-dev-registre-conformite` | block / ask | Partielle |
| 12, 19, 26(6) | Journalisation, conservation | `aiact-art12-journaux-integres`, `aiact-art12-trace-par-decision`, `aiact-art12-trace-de-cloture` | stop / block | Forte |
| 14 | Contrôle humain | `aiact-art14-revue-humaine-par-dossier`, `aiact-art14-confirmation-humaine`, `aiact-art14-plafond-automatisation`, `aiact-art14-garde-fous-intouchables`, `aiact-art14-pas-de-reset-du-garde` | block / ask / stop | Forte (si `human_review` implique un humain) |
| 15 | Exactitude, robustesse | `aiact-dev-evaluations-avant-publication` | block | Partielle |
| 26(11), 86 | Information de la personne | `aiact-art26-11-information-personne` | block | Forte |
| 27 | FRIA | `aiact-art27-fria-prealable` | block | Partielle |
| 50(1) | Divulgation de l’interaction IA | `aiact-art50-1-divulgation-ia` | block | Forte |
| 50(2) | Marquage lisible par machine | `aiact-art50-2-marquage-avant-diffusion`, `aiact-dev-tests-transparence` | block | Forte |
| 50(3) | Information des personnes exposées | `aiact-art50-3-information-personnes-exposees` | block | Forte |
| 50(4) | Hypertrucages, textes d’intérêt public | `aiact-art50-4-hypertrucage-etiquete`, `aiact-art50-4-texte-interet-public` | ask / block | Forte |
| 53(1)(a)(b) | Documentation GPAI | `aiact-art53-1a-documentation-a-jour` | block | Partielle |
| 53(1)(c) | Opt-out droit d’auteur | `aiact-art53-1c-respect-optout-tdm` | block | Partielle (+ refus côté serveur) |
| 53(1)(d) | Résumé des données d’entraînement | `aiact-art53-1d-resume-donnees` | block | Partielle |
| 55 | Évaluations, incidents (risque systémique) | `aiact-art55-evaluations-adversariales`, `aiact-art55-signalement-incident` | block | Partielle |
| 73 | Signalement d’incident grave, gel des preuves | `aiact-art73-signalement-avant-cloture`, `aiact-art73-signalement-rapide`, `aiact-art73-6-preservation-preuves` | block / stop | Forte |
| 4, 17, 18, 43, 49 | Littératie, SMQ, conservation docs, évaluation de conformité, enregistrement | — | — | Hors (plan d’action §10) |

---

## 7. Les rulebooks (AGENTLTL.yaml)

Les modules sont indépendants. Pour un projet, on concatène dans un seul `AGENTLTL.yaml` le socle (00) et les modules pertinents (un seul bloc `settings`, des listes `use` et `rules` fusionnées). Tous ont été compilés avec `agentltl validate` (agentltl-coding 0.3.0) et testés avec `agentltl check` (§9).

| Module | Fichier | Pour qui | Priorité |
|---|---|---|---|
| 00 | `00-socle.yaml` | Tous | **Immédiate** |
| 01 | `01-pratiques-interdites.yaml` | Tout agent ayant des outils de vision, biométrie, génération | **Immédiate** |
| 02 | `02-transparence.yaml` | Agents qui parlent à des personnes ou publient des contenus | **Immédiate** |
| 03 | `03-haut-risque.yaml` | Agents dans un cas d’usage annexe III (RH, crédit, éducation…) | Rodage dès maintenant, blocage avant 12/2027 |
| 04 | `04-donnees.yaml` | Équipes qui entraînent ou affinent des modèles | Haute |
| 05 | `05-gpai.yaml` | Fournisseurs de modèles à usage général | **Immédiate** si concerné |
| 06 | `06-incidents.yaml` | Fournisseurs de systèmes à haut risque, GPAI à risque systémique | Haute |
| 07 | `07-equipe-dev.yaml` | Agents de code d’une équipe qui construit un système d’IA | Haute |

### 7.0 Module 00 — Socle commun

```yaml
# ─────────────────────────────────────────────────────────────────────
# AI Act · Module 00 — Socle commun (à inclure dans tous les projets)
# Couvre : art. 12, 19, 26(6) (journaux), art. 14 (contrôle humain,
# intégrité du dispositif de contrôle), art. 4 (maîtrise de l'IA : les
# rules sont annoncées à l'agent avec leur justification).
# ─────────────────────────────────────────────────────────────────────
settings:
  mode: block                 # par défaut : refus systématique
  retries: 3
  retry_counting: cumulative
  report: 3
  finish_retries: 2
  unparseable:                # une commande non analysable n'est pas contrôlable
    interactive: ask          # → l'humain décide
    auto: deny                # → en mode autonome, refus
  announce: true              # rappel des règles au démarrage et après compaction
  scope: session
  scan_output: true           # alerte si une sortie contient un secret

use:                          # règles packagées de la bibliothèque AgentLTL
  - theme-human-in-the-loop   # publier, envoyer, supprimer à distance → humain
  - theme-data-safety         # secrets, exfiltration de fichiers de données

rules:
  - id: aiact-art12-journaux-integres
    never: [Edit, Write, rm, mv, sed, tee, truncate, shred]
    where: {"*": ["**/logs/audit/**", "**/*.audit.jsonl"]}
    why: >-
      Art. 12, 19 et 26(6) AI Act — les journaux d'un système à haut risque
      sont la preuve de son fonctionnement ; ils doivent rester intègres et
      être conservés (au moins six mois côté déployeur).
    fix: >-
      N'écrivez jamais dans les journaux d'audit : ils sont alimentés par la
      plateforme. Pour tracer une action, appelez l'outil log_event.
    mode: stop

  - id: aiact-art14-garde-fous-intouchables
    never: [Edit, Write, rm, mv, sed, tee]
    where: {"*": ["**/AGENTLTL.yaml", "**/.claude/settings*.json", "**/.agentltl/**"]}
    why: >-
      Art. 14 AI Act — le dispositif de contrôle humain ne doit pas pouvoir
      être désactivé par le système qu'il contrôle.
    fix: Signalez à l'humain la règle qui vous bloque ; lui seul peut la modifier.
    mode: stop

  - id: aiact-art14-pas-de-reset-du-garde
    never: {tool: agentltl, where: {argv: [reset, disable, unuse]}}
    why: >-
      Art. 14 AI Act — effacer la mémoire du garde ou désactiver une règle
      revient à contourner le contrôle humain.
    mode: stop
```

### 7.1 Module 01 — Pratiques interdites (art. 5)

```yaml
# ─────────────────────────────────────────────────────────────────────
# AI Act · Module 01 — Pratiques interdites (art. 5)
# Applicable depuis le 2 février 2025 ; nouvelle interdiction
# « contenus intimes non consentis / CSAM » applicable le 2 décembre 2026
# (Omnibus numérique, règlement (UE) 2026/1744).
# Noms d'outils : noms logiques, à faire correspondre aux outils réels
# (MCP : mcp__<serveur>__<outil>) — voir la table de correspondance.
# ─────────────────────────────────────────────────────────────────────
rules:
  - id: aiact-art5-1c-notation-sociale
    never: ["mcp__*__social_score", "mcp__*__compute_social_score", "mcp__*__rank_citizens"]
    why: >-
      Art. 5(1)(c) — la notation sociale de personnes conduisant à un
      traitement défavorable est interdite.
    mode: stop

  - id: aiact-art5-1e-moissonnage-facial
    never:
      - {tool: ["mcp__*__face_index", "mcp__*__build_face_database", "mcp__*__face_embed_bulk"],
         where: {source: ["web*", "http*", "crawl*", "cctv*", "video_surveillance*"]}}
    why: >-
      Art. 5(1)(e) — constituer ou étendre une base de reconnaissance faciale
      par moissonnage non ciblé d'images d'internet ou de vidéosurveillance
      est interdit.
    mode: stop

  - id: aiact-art5-1f-contexte-declare
    require: {tool: ["mcp__*__emotion_recognition", "mcp__*__detect_emotion"], where: {context: "?*"}}
    why: >-
      Art. 5(1)(f) — l'interdiction dépend du contexte d'usage ; tout appel
      de reconnaissance d'émotions doit déclarer son contexte.
    fix: Ajoutez l'argument context (ex. medical, safety, marketing...).

  - id: aiact-art5-1f-emotions-travail-education
    never:
      - {tool: ["mcp__*__emotion_recognition", "mcp__*__detect_emotion"],
         where: {context: ["workplace", "employ*", "hr*", "recruit*", "education", "school*", "exam*", "student*"]}}
    why: >-
      Art. 5(1)(f) — inférer les émotions d'une personne sur le lieu de
      travail ou dans l'enseignement est interdit (hors raisons médicales ou
      de sécurité, qui relèvent d'une décision humaine documentée).
    mode: stop

  - id: aiact-art5-1g-categorisation-biometrique
    never:
      - {tool: ["mcp__*__biometric_categorize", "mcp__*__infer_attributes_from_face"],
         where: {attribute: ["race*", "ethnic*", "politic*", "union*", "relig*", "belief*", "philosoph*", "sex_life*", "sexual_orient*"]}}
    why: >-
      Art. 5(1)(g) — déduire de données biométriques l'origine, les opinions,
      l'appartenance syndicale, les convictions ou l'orientation sexuelle est
      interdit.
    mode: stop

  - id: aiact-art5-1h-identification-temps-reel
    never: {tool: "mcp__*__remote_biometric_identify", with: {realtime: true}}
    why: >-
      Art. 5(1)(h) — l'identification biométrique à distance en temps réel
      dans l'espace public est interdite, sauf exceptions répressives
      soumises à autorisation judiciaire : jamais à l'initiative d'un agent.
    mode: stop

  - id: aiact-art5-1d-prediction-penale-profilage
    never:
      - {tool: ["mcp__*__predict_offence_risk", "mcp__*__crime_risk_score"],
         where: {basis: ["profil*", "personality*", "traits*"]}}
    why: >-
      Art. 5(1)(d) — évaluer le risque qu'une personne commette une infraction
      sur la seule base de son profilage ou de ses traits de personnalité est
      interdit.
    mode: stop

  - id: aiact-art5-ncii-csam-controle-prealable
    before:
      first: {tool: mcp__aiact__content_safety_check, with: {request_id: $r}}
      then:  {tool: ["mcp__*__generate_image", "mcp__*__generate_video", "mcp__*__edit_image", "mcp__*__edit_video"], with: {request_id: $r}}
    why: >-
      Art. 5 (ajout Omnibus, applicable le 2 décembre 2026) — générer des
      contenus intimes non consentis ou des contenus pédopornographiques est
      interdit ; chaque demande de génération passe d'abord par le contrôle
      de sécurité.
    fix: Appelez mcp__aiact__content_safety_check avec le même request_id, puis générez.
    mode: stop
```

### 7.2 Module 02 — Transparence (art. 50)

```yaml
# ─────────────────────────────────────────────────────────────────────
# AI Act · Module 02 — Transparence (art. 50)
# Applicable depuis le 2 août 2026. Marquage art. 50(2) : 2 décembre 2026
# pour les systèmes génératifs déjà sur le marché avant le 2 août 2026.
# ─────────────────────────────────────────────────────────────────────
rules:
  - id: aiact-art50-1-divulgation-ia
    before:
      first: {tool: mcp__aiact__disclose_ai_interaction, with: {recipient: $r}}
      then:
        - {tool: ["mcp__*__send_message", "mcp__*__post_reply", "mcp__*__start_call"], with: {recipient: $r}}
    scope: project            # une divulgation par interlocuteur, même d'une session à l'autre
    why: >-
      Art. 50(1) — une personne qui interagit avec un système d'IA doit en
      être informée au plus tard lors de la première interaction.
    fix: Appelez mcp__aiact__disclose_ai_interaction pour ce destinataire, puis envoyez.

  - id: aiact-art50-2-marquage-avant-diffusion
    before:
      first: {tool: mcp__aiact__mark_synthetic_content, with: {asset_id: $a}}
      then:  {tool: ["mcp__*__publish_content", "mcp__*__export_asset", "mcp__*__upload_media"], with: {asset_id: $a}}
    scope: project
    why: >-
      Art. 50(2) — les contenus audio, image, vidéo et texte générés doivent
      porter un marquage lisible par machine qui permet de détecter leur
      origine artificielle.
    fix: Appelez mcp__aiact__mark_synthetic_content sur cet asset_id, puis diffusez.

  - id: aiact-art50-4-hypertrucage-etiquete
    require: {tool: "mcp__*__publish_media", with: {ai_disclosure: true}}
    why: >-
      Art. 50(4) — un hypertrucage (image, audio, vidéo réalistes générés ou
      manipulés) doit être signalé comme tel au public.
    fix: Republiez avec ai_disclosure=true (mention visible « généré par IA »).
    mode: ask                 # l'humain tranche les cas artistiques / satiriques

  - id: aiact-art50-4-texte-interet-public
    before:
      first: {tool: mcp__aiact__human_editorial_review, with: {doc_id: $d}}
      then:  {tool: "mcp__*__publish_article", with: {doc_id: $d, ai_disclosure: false}}
    scope: project
    why: >-
      Art. 50(4) — un texte généré publié pour informer le public sur des
      questions d'intérêt public doit être signalé comme généré, sauf s'il a
      fait l'objet d'une revue éditoriale humaine sous responsabilité
      éditoriale.
    fix: Publiez avec ai_disclosure=true, ou faites d'abord valider le texte par mcp__aiact__human_editorial_review.

  - id: aiact-art50-3-information-personnes-exposees
    before:
      first: {tool: mcp__aiact__inform_exposed_persons, with: {session_id: $s}}
      then:  {tool: ["mcp__*__emotion_recognition", "mcp__*__detect_emotion", "mcp__*__biometric_categorize"], with: {session_id: $s}}
    scope: project
    why: >-
      Art. 50(3) — les personnes exposées à un système de reconnaissance des
      émotions ou de catégorisation biométrique doivent en être informées.
```

### 7.3 Module 03 — Systèmes à haut risque

```yaml
# ─────────────────────────────────────────────────────────────────────
# AI Act · Module 03 — Systèmes à haut risque (art. 6, 9, 12, 14, 26, 27)
# Annexe III (emploi, crédit, éducation, biométrie, infrastructures...) :
# applicable le 2 décembre 2027 ; annexe I (produits réglementés) :
# 2 août 2028 (Omnibus). À activer dès maintenant pour constituer la preuve.
# ─────────────────────────────────────────────────────────────────────
rules:
  - id: aiact-art6-classification-prealable
    before:
      first: {tool: mcp__aiact__classify, with: {system: $s}}
      then:
        - {tool: "mcp__*__deploy_system", with: {system: $s}}
        - {tool: "mcp__*__enable_use_case", with: {system: $s}}
    scope: project
    why: >-
      Art. 6 et annexe III — on ne met pas en service un cas d'usage sans
      avoir déterminé s'il est à haut risque : toutes les autres obligations
      en découlent.
    fix: Lancez mcp__aiact__classify pour ce système, puis redemandez le déploiement.

  - id: aiact-art27-fria-prealable
    before:
      first: {tool: mcp__aiact__fria, with: {system: $s}}
      then:  {tool: "mcp__*__enable_use_case", with: {system: $s}}
    scope: project
    why: >-
      Art. 27 — certains déployeurs (organismes publics, services publics,
      crédit et assurance-vie/santé) doivent réaliser une analyse d'impact
      sur les droits fondamentaux avant la première utilisation.
    fix: Réalisez mcp__aiact__fria (elle peut s'appuyer sur l'AIPD RGPD).

  - id: aiact-art9-revue-risques-avant-changement
    before:
      first: {tool: mcp__aiact__risk_review, with: {system: $s}}
      then:  {tool: ["mcp__*__update_model_version", "mcp__*__change_prompt_policy"], with: {system: $s}}
    scope: project
    why: >-
      Art. 9 — le système de gestion des risques couvre tout le cycle de vie :
      un changement de modèle ou de politique se fait après revue des risques.

  - id: aiact-art14-revue-humaine-par-dossier
    before:
      first: {tool: mcp__aiact__human_review, with: {case_id: $c}}
      then:  {tool: ["mcp__*__apply_decision", "mcp__*__notify_decision", "mcp__*__reject_application", "mcp__*__terminate_contract"], with: {case_id: $c}}
    scope: project
    why: >-
      Art. 14 et art. 26(2) — une décision produisant des effets sur une
      personne est soumise à un contrôle humain effectif, dossier par
      dossier ; la recommandation de l'IA ne vaut pas décision.
    fix: Soumettez le dossier via mcp__aiact__human_review et attendez sa validation.

  - id: aiact-art14-confirmation-humaine
    never: ["mcp__*__apply_decision", "mcp__*__reject_application", "mcp__*__terminate_contract"]
    why: >-
      Art. 14(4) — l'humain doit pouvoir décider de ne pas suivre le système
      et l'interrompre ; chaque décision conséquente lui est présentée.
    mode: ask

  - id: aiact-art26-11-information-personne
    before:
      first: {tool: mcp__aiact__inform_affected_person, with: {person_id: $p}}
      then:  {tool: "mcp__*__apply_decision", with: {person_id: $p}}
    scope: project
    why: >-
      Art. 26(11) et art. 86 — la personne concernée par une décision fondée
      sur un système à haut risque doit être informée de son utilisation, et
      peut obtenir une explication.

  - id: aiact-art12-trace-par-decision
    before:
      first: {tool: mcp__aiact__log_event, with: {case_id: $c}}
      then:  {tool: "mcp__*__apply_decision", with: {case_id: $c}}
    why: >-
      Art. 12 — la journalisation doit permettre de retracer chaque
      décision (données d'entrée, recommandation, vérificateur humain).

  - id: aiact-art12-trace-de-cloture
    finally: {call: mcp__aiact__log_event, since: ["mcp__*__apply_decision", "mcp__*__notify_decision"]}
    why: >-
      Art. 12 — le résultat final de chaque décision doit être journalisé
      avant de rendre la main.
    fix: Appelez mcp__aiact__log_event avec le résultat de la dernière décision.

  - id: aiact-art14-plafond-automatisation
    at_most: {call: "mcp__*__apply_decision", times: 20}
    why: >-
      Art. 14(4)(b) — prévenir le biais d'automatisation : au-delà de
      20 décisions dans une session, un humain vérifie la série.
    mode: ask
```

### 7.4 Module 04 — Données et biais

```yaml
# ─────────────────────────────────────────────────────────────────────
# AI Act · Module 04 — Données et biais (art. 10 ; art. 4a introduit
# par l'Omnibus pour le traitement de données sensibles à des fins de
# détection et de correction des biais)
# ─────────────────────────────────────────────────────────────────────
rules:
  - id: aiact-art10-fiche-donnees
    before:
      first: {tool: mcp__aiact__datasheet, with: {dataset: $d}}
      then:  {tool: ["mcp__*__train_model", "mcp__*__finetune_model"], with: {dataset: $d}}
    scope: project
    why: >-
      Art. 10(2) — origine, finalité de collecte, préparation et hypothèses
      des données d'entraînement doivent être documentées avant usage.

  - id: aiact-art10-examen-biais
    before:
      first: {tool: mcp__aiact__bias_examination, with: {dataset: $d}}
      then:  {tool: ["mcp__*__train_model", "mcp__*__finetune_model"], with: {dataset: $d}}
    scope: project
    why: >-
      Art. 10(2)(f)-(g) — les jeux de données sont examinés au regard des
      biais susceptibles d'affecter la santé, la sécurité ou les droits
      fondamentaux, avant l'entraînement.

  - id: aiact-art4a-donnees-sensibles
    never:
      - {tool: ["mcp__*__load_dataset", "mcp__*__query_data"],
         where: {category: ["health*", "ethnic*", "racial*", "biometric*", "genetic*", "relig*", "politic*", "sexual*", "union*"]}}
    why: >-
      Art. 4a (Omnibus) et RGPD art. 9 — les catégories particulières de
      données ne sont traitées que pour détecter ou corriger des biais, avec
      garanties ; chaque accès est approuvé par un humain.
    mode: ask

  - id: aiact-art4a-suppression-apres-usage
    finally:
      call:  {tool: "mcp__*__delete_dataset", where: {category: ["health*", "ethnic*", "racial*", "biometric*", "genetic*", "relig*", "politic*", "sexual*", "union*"]}}
      since: {tool: "mcp__*__load_dataset",   where: {category: ["health*", "ethnic*", "racial*", "biometric*", "genetic*", "relig*", "politic*", "sexual*", "union*"]}}
    why: >-
      Art. 4a (Omnibus) — les données sensibles utilisées pour corriger un
      biais sont supprimées une fois l'objectif atteint.
    fix: Supprimez la copie de travail des données sensibles avant de terminer.
```

### 7.5 Module 05 — Fournisseurs de GPAI

```yaml
# ─────────────────────────────────────────────────────────────────────
# AI Act · Module 05 — Fournisseurs de modèles d'IA à usage général
# (art. 53 ; art. 55 pour les modèles à risque systémique)
# Applicable depuis le 2 août 2025 ; pouvoirs de sanction de la
# Commission depuis le 2 août 2026.
# ─────────────────────────────────────────────────────────────────────
rules:
  - id: aiact-art53-1c-respect-optout-tdm
    before:
      first: {tool: mcp__aiact__check_tdm_optout, with: {domain: $d}}
      then:  {tool: ["mcp__*__crawl", "mcp__*__fetch_for_training"], with: {domain: $d}}
    scope: project
    why: >-
      Art. 53(1)(c) et directive (UE) 2019/790 art. 4(3) — la politique
      droit d'auteur du fournisseur doit respecter les réserves de droits
      (opt-out) exprimées par les titulaires.
    fix: Vérifiez l'opt-out du domaine (robots.txt, TDMRep, métadonnées) avant de collecter.

  - id: aiact-art53-1d-resume-donnees
    before:
      - {tool: mcp__aiact__training_summary, succeeded: true}
      - "mcp__*__release_model"
    scope: project
    why: >-
      Art. 53(1)(d) — un résumé suffisamment détaillé du contenu
      d'entraînement, selon le modèle de l'AI Office, est publié avec le modèle.

  - id: aiact-art53-1a-documentation-a-jour
    before:
      first: {tool: mcp__aiact__doccheck, succeeded: true}
      then:  "mcp__*__release_model"
      since: ["mcp__*__train_model", "mcp__*__finetune_model"]
    why: >-
      Art. 53(1)(a)-(b) — la documentation technique et l'information des
      fournisseurs en aval doivent décrire le modèle effectivement publié.
    fix: Mettez à jour la documentation et relancez mcp__aiact__doccheck après le dernier entraînement.

  - id: aiact-art55-evaluations-adversariales
    before:
      first: {tool: mcp__aiact__run_adversarial_evals, succeeded: true}
      then:  "mcp__*__release_model"
      since: ["mcp__*__train_model", "mcp__*__finetune_model"]
    why: >-
      Art. 55(1)(a) — un modèle à risque systémique est évalué, y compris par
      des tests adversariaux, sur la version qui sera publiée.

  - id: aiact-art55-signalement-incident
    finally: {call: mcp__aiact__report_incident, since: mcp__aiact__flag_serious_incident}
    why: >-
      Art. 55(1)(c) — les incidents graves sont suivis, documentés et
      signalés sans retard injustifié à l'AI Office.
```

### 7.6 Module 06 — Incidents graves

```yaml
# ─────────────────────────────────────────────────────────────────────
# AI Act · Module 06 — Incidents graves et surveillance (art. 72, 73)
# ─────────────────────────────────────────────────────────────────────
rules:
  - id: aiact-art73-signalement-avant-cloture
    finally: {call: mcp__aiact__report_serious_incident, since: mcp__aiact__flag_serious_incident}
    why: >-
      Art. 73 — un incident grave est notifié à l'autorité de surveillance
      (au plus tard 15 jours, 10 jours en cas de décès, 2 jours en cas
      d'atteinte généralisée ou d'infrastructure critique).
    fix: Appelez mcp__aiact__report_serious_incident (il prévient le responsable conformité).

  - id: aiact-art73-signalement-rapide
    ltl: 'within_steps("mcp__aiact__flag_serious_incident", "mcp__aiact__report_serious_incident", 5)'
    why: >-
      Art. 73 — le signalement passe avant toute autre action : au plus
      cinq appels d'outil entre la détection et le signalement.

  - id: aiact-art73-6-preservation-preuves
    ltl: 'G((now("mcp__mlops__rollback_model") | now("mcp__mlops__update_model_version") | now("mcp__mlops__delete_model_artifact")) -> (((!now("mcp__aiact__flag_serious_incident")) S now("mcp__aiact__report_serious_incident")) | H(!now("mcp__aiact__flag_serious_incident"))))'
    scope: project
    why: >-
      Art. 73(6) — après un incident grave, on ne modifie pas le système
      d'une manière qui fausserait l'enquête avant d'en avoir informé
      l'autorité.
    fix: Signalez d'abord l'incident (mcp__aiact__report_serious_incident).
    mode: stop
```

### 7.7 Module 07 — Agents de code (Claude Code, Copilot CLI, Vibe, Codex)

```yaml
# ─────────────────────────────────────────────────────────────────────
# AI Act · Module 07 — Agents de code (Claude Code, Copilot CLI,
# Mistral Vibe, Codex) dans une équipe qui construit un système d'IA.
# S'appuie sur une petite CLI interne « aiact » (à créer, voir plan),
# décrite ci-dessous pour que les règles puissent nommer ses arguments.
# ─────────────────────────────────────────────────────────────────────
use:
  - theme-release-safety      # tests avant push, plan avant apply, prod → humain

rules:
  - id: aiact-dev-classification-avant-prod
    before:
      first: {tool: aiact_classify, with: {system: $s}}
      then:  {tool: helm_upgrade, with: {release: $s, namespace: prod}}
    scope: project
    why: >-
      Art. 6 — un système ne part pas en production sans classification de
      risque enregistrée pour ce même système.
    fix: Lancez `aiact classify --system <release>` puis redéployez.

  - id: aiact-dev-doc-technique-a-jour
    before:
      first: {tool: aiact_doccheck, succeeded: true}
      then:  {tool: helm_upgrade, with: {namespace: prod}}
      since: {tool: [Edit, Write], where: {file_path: ["src/**", "models/**", "prompts/**"]}}
    why: >-
      Art. 11 et annexe IV — la documentation technique décrit la version
      déployée ; tout changement de code, de modèle ou de prompt la rend
      obsolète.
    fix: Mettez à jour docs/annexe-iv puis lancez `aiact doccheck` (il doit réussir).

  - id: aiact-dev-evaluations-avant-publication
    before:
      first: {tool: aiact_evals, succeeded: true}
      then:
        - {tool: helm_upgrade, with: {namespace: prod}}
        - {tool: huggingface-cli, with: {argv: upload}}
        - {tool: hf, with: {argv: upload}}
      since: {tool: [Edit, Write], where: {file_path: ["src/**", "models/**", "prompts/**"]}}
    why: >-
      Art. 15 — exactitude, robustesse et cybersécurité sont mesurées sur la
      version publiée (suite d'évaluation, biais, red-teaming).
    fix: Lancez `aiact evals` après votre dernière modification.

  - id: aiact-dev-fiche-donnees-avant-entrainement
    before:
      first: {tool: aiact_datasheet, succeeded: true}
      then:  {tool: [python, python3], where: {script: ["train*.py", "**/train*.py", "finetune*.py", "**/finetune*.py"]}}
      since: {tool: [Edit, Write], where: {file_path: ["data/**", "datasets/**"]}}
    why: >-
      Art. 10 — les données d'entraînement modifiées sont documentées avant
      tout entraînement.

  - id: aiact-dev-registre-conformite
    never: [Edit, Write, rm, mv, sed, tee]
    where: {"*": ["compliance/**", "docs/annexe-iv/**"]}
    why: >-
      Art. 11, 17 et 18 — le registre de classification, la FRIA et la
      documentation technique sont des pièces du système de gestion de la
      qualité ; leur modification est validée par un humain.
    mode: ask

  - id: aiact-dev-bibliotheques-biometriques
    never:
      - {tool: pip_install, where: {"*": ["deepface*", "face_recognition*", "fer", "fer==*", "insightface*", "py-feat*", "hsemotion*"]}}
      - {tool: [npm, yarn, pnpm], where: {argv: ["face-api*", "@vladmandic/face-api*", "@vladmandic/human*"]}}
    why: >-
      Art. 5(1)(e)-(g) et annexe III(1) — la biométrie et la reconnaissance
      d'émotions sont interdites ou à haut risque selon le contexte ; leur
      introduction dans le code est une décision humaine.
    mode: ask

  - id: aiact-dev-tests-transparence
    finally:
      call:  {tool: pytest, succeeded: true}
      since: {tool: [Edit, Write], where: {file_path: ["**/*disclos*", "**/*watermark*", "**/*marking*", "**/*c2pa*"]}}
    why: >-
      Art. 50 — le code de divulgation et de marquage est une obligation
      légale : on ne rend pas la main sans que ses tests passent.
    fix: Lancez pytest et corrigez jusqu'au vert.

tools:
  aiact:
    subcommands:
      classify:  {options: [{flags: [--system]}]}
      fria:      {options: [{flags: [--system]}]}
      doccheck:  {options: [{flags: [--system]}]}
      evals:     {options: [{flags: [--system]}, {flags: [--suite]}]}
      datasheet: {options: [{flags: [--dataset]}]}
      mark:      {options: [{flags: [--asset]}]}
  helm:
    options:
      - {flags: [-n, --namespace], global: true}
    subcommands:
      upgrade: {positionals: [{name: release}, {name: chart}]}
```

**Note sur `theme-human-in-the-loop` (module 00).** Ce thème de la bibliothèque demande l’accord de l’humain avant *chaque* envoi de message ou publication. Il s’ajoute aux règles AI Act : après divulgation, un envoi passe en `ask` au lieu d’être refusé. C’est un choix prudent pour la phase de lancement ; pour un agent conversationnel à fort volume, remplacez-le par `aiact-art14-plafond-automatisation` adapté aux messages et une revue par échantillonnage.

---

## 8. Variante Python pour les agents métier

Pour un agent maison (smolagents, boucle OpenAI-compatible, LangChain via `Enforcer`), les mêmes obligations s’expriment avec la bibliothèque `agentltl`. Ce fichier a été exécuté : la décision sans revue est bloquée, la décision après revue passe, la décision suivante sans nouvelle revue est bloquée, et la trace conforme obtient un score de 1.0.

```python
"""AI Act · contraintes AgentLTL pour un agent métier (boucle d'agent quelconque)."""
from agentltl import (Called, Constraint, ConstraintSeverity as S, Enforcer, Implies,
                      WithinSteps, parse, verify_trace)

CONSTRAINTS = [
    # Art. 5(1)(c) — notation sociale : jamais
    Constraint("art5-notation-sociale", parse('G(!now("social_score"))'),
               repair="La notation sociale est interdite (art. 5(1)(c))."),
    # Art. 14 — chaque décision exige une revue humaine depuis la décision précédente
    Constraint("art14-une-revue-par-decision",
               parse('G(now("apply_decision") -> Y((!now("apply_decision")) S now("human_review")))'),
               repair="Soumettez ce dossier à human_review avant d'appliquer la décision."),
    # Art. 50(1) — divulgation avant tout premier message
    Constraint("art50-divulgation",
               parse('G(now("send_message") -> O(now("disclose_ai_interaction")))'),
               repair="Informez l'interlocuteur qu'il échange avec une IA."),
    # Art. 73 — signalement au plus 5 appels après la détection
    Constraint("art73-signalement",
               Implies(Called("flag_serious_incident"),
                       WithinSteps("flag_serious_incident", "report_serious_incident", 5)),
               repair="Signalez l'incident grave avant toute autre action."),
]
SEVERITIES = {
    "art5-notation-sociale": S.HARD_STOP,
    "art14-une-revue-par-decision": S.PERSISTENT_BLOCK,
    "art50-divulgation": S.PERSISTENT_BLOCK,
    "art73-signalement": S.HARD_STOP,
}

def make_enforcer() -> Enforcer:
    return Enforcer(CONSTRAINTS, SEVERITIES)

if __name__ == "__main__":
    enf = make_enforcer()
    for tool, args in [("apply_decision", {"case_id": "C1"}),
                       ("human_review", {"case_id": "C1"}),
                       ("apply_decision", {"case_id": "C1"}),
                       ("apply_decision", {"case_id": "C2"})]:
        d = enf.check(tool, args)
        print(f"{tool:16} {args} -> {d.action if not d.allowed else 'allow'}")
        if d.allowed:
            enf.record_completed(tool, args, result="ok", status=0)
    # Preuve d'audit a posteriori : score de conformité de la trace
    trace = {"tool_calls": [{"tool_name": "human_review"}, {"tool_name": "apply_decision"}]}
    report = verify_trace(trace, CONSTRAINTS)
    print(report["compliance_label"], report["compliance_score"])
```

Atouts de la variante Python : `CalledWithResult` permet d’exiger que `check_tdm_optout` ait *renvoyé* « autorisé » (et pas seulement qu’il ait été appelé), et `verify_trace` fournit un **score de conformité pondéré** de chaque trace, utilisable comme preuve d’audit et comme indicateur de suivi.

---

## 9. Tests et validation

### 9.1 Validation statique

```bash
pip install agentltl-coding
agentltl validate AGENTLTL.yaml        # compile, liste les règles, signale les cibles mortes
agentltl translate "helm upgrade web repo/web -n prod"   # vérifie les noms d’arguments
agentltl tools "aiact_*"               # outils et arguments que les règles peuvent viser
```

### 9.2 Tests de non-régression

Chaque scénario rejoue une séquence d’appels et vérifie la décision attendue (`allow`, `deny`, `ask`, `stop`). À intégrer à la CI du dépôt qui porte `AGENTLTL.yaml` : toute modification d’une règle doit les faire passer.

```bash
#!/usr/bin/env bash
# Scénarios de non-régression des rulebooks AI Act (agentltl check).
# Usage : depuis le dépôt contenant AGENTLTL.yaml →  bash tests-aiact.sh
set -u
R="--rules AGENTLTL.yaml"
fail=0
run() { echo "▶ $1"; shift; agentltl check $R "$@" >/tmp/aiact-check.log 2>&1 || { fail=1; cat /tmp/aiact-check.log; }; }

run "Art. 50(2) : pas de diffusion sans marquage du même contenu" \
  'deny: mcp__cms__upload_media {"asset_id": "img-42"}' \
  'allow: mcp__aiact__mark_synthetic_content {"asset_id": "img-42"}' \
  'deny: mcp__cms__upload_media {"asset_id": "img-43"}'

run "Art. 14 / 12 / 26(11) : décision = information + journal + revue humaine du même dossier" \
  'deny: mcp__rh__apply_decision {"case_id": "C1", "person_id": "P1"}' \
  'allow: mcp__aiact__inform_affected_person {"person_id": "P1"}' \
  'allow: mcp__aiact__log_event {"case_id": "C1"}' \
  'deny: mcp__rh__apply_decision {"case_id": "C1", "person_id": "P1"}' \
  'allow: mcp__aiact__human_review {"case_id": "C1"}'

run "Art. 5 (Omnibus) : génération d'image sans contrôle de sécurité de la même requête" \
  'stop: mcp__studio__generate_image {"request_id": "R9", "prompt": "..."}'

run "Art. 5(1)(e) : moissonnage facial depuis le web" \
  'stop: mcp__vision__face_index {"source": "web_crawl", "target": "db"}'

run "Art. 53(1)(c) : opt-out TDM vérifié pour le même domaine" \
  'deny: mcp__data__crawl {"domain": "example.org"}' \
  'allow: mcp__aiact__check_tdm_optout {"domain": "example.org"}' \
  'allow: mcp__data__crawl {"domain": "example.org"}'

run "Art. 73(6) : pas de rollback entre détection et signalement d'un incident grave" \
  'allow: mcp__aiact__flag_serious_incident {"id": "INC-1"}' \
  'stop: mcp__mlops__rollback_model {"version": "1.4"}'

run "Socle : l'agent ne peut pas modifier ses propres règles ni les journaux" \
  'stop: Edit {"file_path": "AGENTLTL.yaml", "old_string": "a", "new_string": "b"}' \
  'stop: rm logs/audit/2026-11.audit.jsonl'

run "Module 07 : prod seulement après classification, doc à jour et évaluations" \
  'deny: helm upgrade cv-screener repo/cv --version 2.0 -n prod' \
  'allow: aiact classify --system cv-screener' \
  'allow: aiact doccheck --system cv-screener' \
  'allow: aiact evals --system cv-screener' \
  'ask: helm upgrade cv-screener repo/cv --version 2.0 -n prod'

run "Module 07 : bibliothèques biométriques et registre de conformité → humain" \
  'ask: pip install deepface' \
  'ask: Edit {"file_path": "compliance/registre.yaml", "old_string": "a", "new_string": "b"}'

exit $fail
```

Résultat sur l’ensemble des modules fusionnés : **9 scénarios sur 9 conformes**.

### 9.3 Tests adversariaux (à mener en phase 2)

| Attaque | Exemple | Défense attendue |
|---|---|---|
| Contournement par le shell | `eval "$(echo rm logs/audit/x)"` | `unparseable: ask/deny` |
| Script opaque | `python publish.py img-42` | Exposer la publication uniquement via l’outil MCP ; retirer l’accès réseau direct |
| Mensonge sur un argument | `detect_emotion {context: "marketing"}` en contexte RH | Contexte injecté par le serveur, pas par l’agent |
| Auto-validation | L’agent appelle `human_review` et s’approuve | `human_review` bloquant, signé par un humain identifié |
| Désactivation du garde | Édition de `AGENTLTL.yaml`, `agentltl disable` | `aiact-art14-garde-fous-intouchables`, `aiact-art14-pas-de-reset-du-garde` |
| Changement d’identifiant | Marquer `img-42`, publier `img-42-final` | Variable liée : refus (testé) |
| Sous-agent | Déléguer l’action à un sous-agent | Vérifier que le harnais applique les règles aux sous-agents |

---

## 10. Plan d’action phasé

### Phase 0 — Cadrage (12 → 19 novembre 2026)

| # | Action | Livrable | Responsable |
|---|---|---|---|
| 0.1 | Inventaire des agents en production et de leurs outils (MCP, CLI, API) | Registre des agents | Tech lead |
| 0.2 | Pour chaque agent : rôle AI Act (fournisseur / déployeur / GPAI) et qualification provisoire | Fiche de qualification | Juriste + DPO |
| 0.3 | Repérer les systèmes génératifs mis sur le marché **avant le 2 août 2026** (échéance 50(2) au 2 décembre) | Liste prioritaire | Produit |
| 0.4 | Table de correspondance noms logiques → noms réels des outils | `mapping.md` | Tech lead |
| 0.5 | Installer AgentLTL dans un environnement de test | Environnement | DevOps |

### Phase 1 — Urgences du 2 décembre 2026 (19 novembre → 1er décembre)

| # | Action | Livrable |
|---|---|---|
| 1.1 | Déployer le module 00 en `block` / `stop` sur tous les agents | Socle actif |
| 1.2 | Déployer le module 01 en `log` 3 jours, analyser les déclenchements, puis `stop` | Interdictions actives |
| 1.3 | Implémenter `content_safety_check` et `mark_synthetic_content` (version minimale) | Outils MCP |
| 1.4 | Déployer le module 02 (`aiact-art50-2-…` en `block` au plus tard le 1er décembre) | Transparence active |
| 1.5 | Lancer `tests-aiact.sh` en CI | Pipeline vert |
| 1.6 | Former les personnes qui reçoivent les demandes `ask` (art. 4) | Support + session de 1 h |

### Phase 2 — Industrialisation (décembre 2026 → février 2027)

| # | Action | Livrable |
|---|---|---|
| 2.1 | Construire le serveur MCP `aiact` complet (§5.1) et la CLI `aiact` (§5.2) | Code + docs |
| 2.2 | Rendre `human_review` bloquant et signé (workflow d’approbation) | Workflow |
| 2.3 | Déplacer les arguments décisifs (`context`, `source`, `category`) côté serveur | Serveurs durcis |
| 2.4 | Déployer modules 04, 05 (si GPAI), 06, 07 en `block` | Modules actifs |
| 2.5 | Campagne de tests adversariaux (§9.3) | Rapport de red-team |
| 2.6 | Diffuser le rulebook via la configuration gérée par l’administrateur (voir la page *Operating it* de la doc AgentLTL) | Configuration centralisée |

### Phase 3 — Haut risque en rodage (T1 → T3 2027)

| # | Action | Livrable |
|---|---|---|
| 3.1 | Activer le module 03 en `log` sur les agents annexe III | Traces de référence |
| 3.2 | Réaliser les FRIA concernées (art. 27), en s’appuyant sur les AIPD | FRIA |
| 3.3 | Mesurer : décisions sans revue, délais de revue, taux d’`ask` | Tableau de bord |
| 3.4 | Basculer progressivement en `block` / `ask` | Module 03 actif |
| 3.5 | Relier les journaux AgentLTL au SMQ (art. 17) et à la documentation technique (annexe IV) | Dossier de conformité |

### Phase 4 — Échéance haut risque (T3 → 2 décembre 2027)

| # | Action | Livrable |
|---|---|---|
| 4.1 | Audit à blanc : rejouer 3 mois de traces avec `verify_trace` | Rapport de conformité |
| 4.2 | Vérifier la conservation des journaux (≥ 6 mois) et leur intégrité | Attestation |
| 4.3 | Évaluation de conformité, enregistrement dans la base UE (art. 43, 49) | Dossiers |
| 4.4 | Pour les produits annexe I : même démarche avant le 2 août 2028 | Plan dédié |

### Phase 5 — Amélioration continue (permanent)

Veille trimestrielle (lignes directrices de la Commission, codes de pratique sur le marquage, normes harmonisées CEN-CENELEC), revue des règles à chaque nouvel outil d’agent, revue annuelle de la matrice de couverture, retour d’expérience après chaque incident.

---

## 11. Gouvernance, preuves et indicateurs

### 11.1 Responsabilités (RACI simplifié)

| Activité | Juriste / DPO | Resp. conformité IA | Tech lead | Équipe produit |
|---|---|---|---|---|
| Qualification des systèmes | **R** | A | C | C |
| Rédaction / modification des règles | C | **A** | R | I |
| Implémentation des outils `aiact` | I | C | **R/A** | C |
| Réponse aux demandes `ask` | I | C | I | **R** (personnes habilitées) |
| Revue trimestrielle | C | **R/A** | C | C |

Toute modification de `AGENTLTL.yaml` passe par une revue de code approuvée par le responsable conformité IA : c’est le pendant organisationnel de la règle `aiact-art14-garde-fous-intouchables`.

### 11.2 Preuves produites

| Preuve | Source | Article |
|---|---|---|
| Trace complète des appels d’outil par session | `agentltl trace` | 12, 26(6) |
| Refus motivés et validations humaines | Journaux du garde + `human_review` | 14 |
| Score de conformité par trace | `verify_trace` | 9, 17 |
| Version du rulebook en vigueur à chaque date | Historique git | 11, 18 |
| Résultats des tests de non-régression | CI | 15, 17 |

### 11.3 Indicateurs

| Indicateur | Cible |
|---|---|
| Décisions à haut risque appliquées sans revue humaine | 0 |
| Contenus diffusés sans marquage (systèmes concernés) | 0 |
| Délai moyen de réponse à une demande `ask` | < 4 h ouvrées |
| Déclenchements `stop` (pratiques interdites) | Suivi, analyse de chaque cas |
| Score de conformité moyen des traces | ≥ 0,98, puis 1,0 |
| Incidents graves signalés dans le délai légal | 100 % |
| Couverture de tests (règles avec ≥ 1 scénario) | 100 % |

---

## 12. Limites et risques résiduels

| Limite | Risque | Mesure compensatoire |
|---|---|---|
| Contrôle procédural, pas sémantique | `classify` appelé mais qualification erronée | Validation juridique humaine dans `classify` |
| Arguments fournis par l’agent | L’agent déclare un faux contexte | Arguments injectés côté serveur (principe 2) |
| Contenu non inspecté | Manipulation (art. 5(1)(a)(b)) non détectée | Classifieurs de contenu, évaluations, politiques du modèle |
| Scripts opaques | Action cachée dans un script | Exposer les actions sensibles uniquement via MCP ; restreindre le réseau |
| Variable + `succeeded` impossible côté `first` | Un contrôle échoué « compte » quand même | Outils de conformité qui *refusent* en cas d’échec ; variante Python `CalledWithResult` |
| `since` + variables non supporté | Doc à jour vérifiée globalement, pas par système | Un dépôt par système, ou CLI `aiact doccheck` multi-systèmes |
| Mémoire projet locale | Flotte d’agents sur plusieurs machines | Centraliser l’état (`Enforcer.to_state()` / `from_state()`) |
| Fatigue d’approbation | Des `ask` validés machinalement | Plafonds, échantillonnage, indicateur de délai et de taux d’approbation |
| Évolution du droit | Lignes directrices, normes, actes délégués | Phase 5 : veille et revue trimestrielle |

---

## 13. Sources

- Règlement (UE) 2024/1689 (AI Act) — texte et explorateur : https://artificialintelligenceact.eu
- Omnibus numérique sur l’IA, règlement (UE) 2026/1744 : https://artificialintelligenceact.eu/ai-act-explorer/digital-omnibus/
- Service Desk AI Act de la Commission : https://ai-act-service-desk.ec.europa.eu/en
- AgentLTL — documentation : https://agentltl.github.io/ ; référence `AGENTLTL.yaml` : https://agentltl.github.io/rules/reference/ ; bibliothèque Python : https://agentltl.github.io/python/
- AgentLTL — article de recherche : https://arxiv.org/abs/2607.02599
- Analyses de l’Omnibus (calendrier et contenu) : Cooley, Gibson Dunn, William Fry, DLA Piper, Goodwin (2026)
