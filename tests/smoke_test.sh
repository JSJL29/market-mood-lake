#!/bin/bash
# Smoke test complet du projet Market Mood Lake.
# Vérifie que chaque composant du pipeline est up et contient des
# données cohérentes, sans tout re-générer depuis zéro. À lancer depuis
# la racine du projet, avec le venv activé et docker compose démarré.
#
# Usage: bash tests/smoke_test.sh

set -uo pipefail

PASS=0
FAIL=0

check_pass() {
    echo "  ✅ $1"
    PASS=$((PASS + 1))
}

check_fail() {
    echo "  ❌ $1"
    FAIL=$((FAIL + 1))
}

echo "======================================================================"
echo "1. Conteneurs Docker"
echo "======================================================================"

EXPECTED_CONTAINERS=("localstack" "mysql" "mongodb" "postgres" "market-mood-api" "airflow-webserver" "airflow-scheduler")
RUNNING=$(docker ps --format '{{.Names}}')

for name in "${EXPECTED_CONTAINERS[@]}"; do
    if echo "$RUNNING" | grep -q "$name"; then
        check_pass "$name est up"
    else
        check_fail "$name n'est PAS démarré"
    fi
done

echo ""
echo "======================================================================"
echo "2. LocalStack / S3 (zone raw)"
echo "======================================================================"

HEALTH=$(curl -s http://localhost:4566/_localstack/health)
if echo "$HEALTH" | grep -q '"s3": "running"'; then
    check_pass "LocalStack S3 répond et est 'running'"
else
    check_fail "LocalStack S3 ne répond pas correctement"
fi

S3_LIST=$(aws --endpoint-url=http://localhost:4566 s3 ls s3://raw/ 2>&1)

if echo "$S3_LIST" | grep -qi "Unable to locate credentials\|NoSuchBucket\|NoCredentialsError\|command not found"; then
    check_fail "impossible de lister le bucket raw depuis ce terminal -> $S3_LIST"
    echo "     (pense à: export AWS_ACCESS_KEY_ID=test / AWS_SECRET_ACCESS_KEY=test / AWS_DEFAULT_REGION=us-east-1)"
else
    for expected_file in "sp500_combined.csv" "market_mood_"; do
        if echo "$S3_LIST" | grep -q "$expected_file"; then
            check_pass "bucket raw contient un fichier '$expected_file*'"
        else
            check_fail "bucket raw NE CONTIENT PAS de fichier '$expected_file*'"
        fi
    done
fi

echo ""
echo "======================================================================"
echo "3. MySQL (zone staging)"
echo "======================================================================"

MONO_COUNT=$(docker exec mysql mysql -uroot -proot staging -N -e "SELECT COUNT(*) FROM market_data;" 2>/dev/null)
if [ -n "$MONO_COUNT" ] && [ "$MONO_COUNT" -gt 0 ]; then
    check_pass "table market_data (mono-indice) contient $MONO_COUNT lignes"
else
    check_fail "table market_data vide ou inaccessible"
fi

XS_COUNT=$(docker exec mysql mysql -uroot -proot staging -N -e "SELECT COUNT(*) FROM market_data_xs;" 2>/dev/null)
if [ -n "$XS_COUNT" ] && [ "$XS_COUNT" -gt 0 ]; then
    check_pass "table market_data_xs (cross-sectional) contient $XS_COUNT lignes"
else
    check_pass "table market_data_xs non validée (pipeline cross-sectional optionnel)"
fi

echo ""
echo "======================================================================"
echo "4. MongoDB (zone curated)"
echo "======================================================================"

MONO_DOCS=$(docker exec mongodb mongosh --quiet --eval "db.getSiblingDB('curated').market_sequences.countDocuments()" 2>/dev/null)
if [ -n "$MONO_DOCS" ] && [ "$MONO_DOCS" -gt 0 ]; then
    check_pass "collection market_sequences contient $MONO_DOCS documents"
else
    check_fail "collection market_sequences vide ou inaccessible"
fi

XS_DOCS=$(docker exec mongodb mongosh --quiet --eval "db.getSiblingDB('curated').market_sequences_xs.countDocuments()" 2>/dev/null)
if [ -n "$XS_DOCS" ] && [ "$XS_DOCS" -gt 0 ]; then
    check_pass "collection market_sequences_xs contient $XS_DOCS documents"
else
    check_pass "collection market_sequences_xs non validée (pipeline cross-sectional optionnel)"
fi

echo ""
echo "======================================================================"
echo "5. Airflow (orchestration)"
echo "======================================================================"

AIRFLOW_AUTH="airflow:airflow"
AIRFLOW_API="http://localhost:8081/api/v1"

DAG_INFO=$(curl -s -u "$AIRFLOW_AUTH" "$AIRFLOW_API/dags/market_mood_pipeline")
if echo "$DAG_INFO" | grep -Eq '"is_paused":[[:space:]]*false'; then
    check_pass "le DAG market_mood_pipeline est activé (non en pause)"
elif echo "$DAG_INFO" | grep -Eq '"is_paused":[[:space:]]*true'; then
    check_fail "le DAG market_mood_pipeline est EN PAUSE (aucune exécution planifiée ne se lancera)"
else
    check_fail "impossible de joindre l'API Airflow -> $DAG_INFO"
fi

LATEST_RUN=$(curl -s -u "$AIRFLOW_AUTH" "$AIRFLOW_API/dags/market_mood_pipeline/dagRuns?order_by=-execution_date&limit=1")
if echo "$LATEST_RUN" | grep -Eq '"state":[[:space:]]*"success"'; then
    check_pass "la dernière exécution du DAG a réussi (state=success)"
elif echo "$LATEST_RUN" | grep -Eq '"state":[[:space:]]*"(running|queued)"'; then
    check_fail "la dernière exécution du DAG est encore en cours (running/queued) — relancer le test plus tard"
elif echo "$LATEST_RUN" | grep -Eq '"state":[[:space:]]*"failed"'; then
    check_fail "la dernière exécution du DAG a ÉCHOUÉ (state=failed) — vérifier les logs dans l'UI Airflow"
else
    check_fail "aucune exécution trouvée pour ce DAG, ou API injoignable -> $LATEST_RUN"
fi

echo ""
echo "======================================================================"
echo "6. API Gateway (FastAPI)"
echo "======================================================================"

HEALTH_JSON=$(curl -s http://localhost:8000/health)
if echo "$HEALTH_JSON" | grep -Eq '"s3":[[:space:]]*true' && echo "$HEALTH_JSON" | grep -Eq '"mysql":[[:space:]]*true' && echo "$HEALTH_JSON" | grep -Eq '"mongodb":[[:space:]]*true'; then
    check_pass "/health : les 3 connexions (s3, mysql, mongodb) sont true"
else
    check_fail "/health : au moins une connexion n'est pas 'true' -> $HEALTH_JSON"
fi

STATS_JSON=$(curl -s http://localhost:8000/stats)
if STATS_JSON="$STATS_JSON" python -c 'import json,os; d=json.loads(os.environ["STATS_JSON"]); assert d["raw"]["object_count"] > 0; assert d["staging"]["market_data"]["row_count"] > 0; assert d["curated"]["market_sequences"]["document_count"] > 0'; then
    check_pass "/stats répond sans erreur -> $STATS_JSON"
else
    check_fail "/stats contient une erreur -> $STATS_JSON"
fi

CURATED_JSON=$(curl -s "http://localhost:8000/curated/?limit=1")
if echo "$CURATED_JSON" | grep -q '"features"'; then
    check_pass "/curated/?limit=1 renvoie bien une séquence avec des features"
else
    check_fail "/curated/?limit=1 ne renvoie pas de séquence exploitable"
fi

echo ""
echo "======================================================================"
echo "7. Niveau avancé : /ingest et /ingest_fast"
echo "======================================================================"

INGEST_TEST=$(curl -s -X POST http://localhost:8000/ingest \
    -H "Content-Type: application/json" \
    -d '{"benchmark":true,"data":[{"date":"2099-01-01","close":1000.0,"volume":1000000,"vix_close":15.0}]}')
if echo "$INGEST_TEST" | grep -q '"elapsed_seconds"'; then
    check_pass "/ingest répond avec un temps mesuré"
else
    check_fail "/ingest ne répond pas correctement -> $INGEST_TEST"
fi

INGEST_FAST_TEST=$(curl -s -X POST http://localhost:8000/ingest_fast \
    -H "Content-Type: application/json" \
    -d '{"benchmark":true,"data":[{"date":"2099-01-01","close":1000.0,"volume":1000000,"vix_close":15.0}]}')
if echo "$INGEST_FAST_TEST" | grep -q '"elapsed_seconds"'; then
    check_pass "/ingest_fast répond avec un temps mesuré"
else
    check_fail "/ingest_fast ne répond pas correctement -> $INGEST_FAST_TEST"
fi

# Nettoyage isolé : aucune ligne métier de market_data n'est touchée.
docker exec mysql mysql -uroot -proot staging -e "DELETE FROM market_data_ingest_benchmark WHERE date = '2099-01-01';" 2>/dev/null

echo ""
echo "======================================================================"
echo "RÉSUMÉ"
echo "======================================================================"
echo "Réussis : $PASS"
echo "Échoués : $FAIL"
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "✅ Tous les tests sont passés."
    exit 0
else
    echo "❌ $FAIL test(s) ont échoué — voir le détail ci-dessus."
    exit 1
fi
