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
