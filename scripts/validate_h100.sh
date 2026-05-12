#!/usr/bin/env bash
# validate_h100.sh — Validación funcional post-deploy en el servidor H100.
#
# Corre 5 checks básicos y 5 preguntas de prueba al agente.
# Requiere que banks-api esté corriendo y que BANKS_API_KEY esté seteada.
#
# Uso:
#   BANKS_API_KEY=<tu-key> bash scripts/validate_h100.sh
#   BANKS_API_KEY=<tu-key> BANKS_API_URL=http://localhost:8080 bash scripts/validate_h100.sh

set -euo pipefail

API_URL="${BANKS_API_URL:-http://localhost:8080}"
API_KEY="${BANKS_API_KEY:-}"
INSTALL_DIR="/opt/banks_rag"
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; ((PASS++)); }
fail() { echo -e "  ${RED}✗${NC} $1"; ((FAIL++)); }
info() { echo -e "  ${YELLOW}→${NC} $1"; }

echo "=== Banks RAG — Validación H100 ==="
echo "API: $API_URL"
echo ""

# ── Check 1: Liveness ─────────────────────────────────────────────────────
echo "1. Liveness (/healthz)"
STATUS=$(curl -sf "$API_URL/healthz" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "ERROR")
if [[ "$STATUS" == "ok" ]]; then
    ok "healthz → $STATUS"
else
    fail "healthz → $STATUS (esperado: ok)"
fi

# ── Check 2: Readiness ────────────────────────────────────────────────────
echo "2. Readiness (/readyz)"
READY=$(curl -sf "$API_URL/readyz" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "ERROR")
if [[ "$READY" == "ok" ]]; then
    ok "readyz → $READY"
elif [[ "$READY" == "loading" ]]; then
    info "readyz → loading (LLM cargando, esperar 30-60s)"
else
    fail "readyz → $READY"
fi

# ── Check 3: Métricas Prometheus ─────────────────────────────────────────
echo "3. Métricas (/metrics)"
METRICS=$(curl -sf "$API_URL/metrics" 2>/dev/null | grep -c "^banks_" || echo "0")
if [[ "$METRICS" -gt 0 ]]; then
    ok "metrics expone $METRICS líneas banks_*"
else
    fail "metrics no expone métricas banks_*"
fi

# ── Check 4: Búsqueda RAG ────────────────────────────────────────────────
echo "4. Búsqueda híbrida (/v1/search)"
if [[ -z "$API_KEY" ]]; then
    info "BANKS_API_KEY no seteada, saltando check autenticado"
else
    HITS=$(curl -sf "$API_URL/v1/search" \
        -H "X-API-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"query": "política monetaria Banco Central", "k": 3}' \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('hits',[])))" 2>/dev/null || echo "0")
    if [[ "$HITS" -gt 0 ]]; then
        ok "search → $HITS hits"
    else
        fail "search → 0 hits (¿BD vacía o error?)"
    fi
fi

# ── Check 5: SQL routing ─────────────────────────────────────────────────
echo "5. Evaluación SQL routing (offline)"
ROUTING=$(cd "$INSTALL_DIR" && .venv/bin/banks-eval routing 2>/dev/null | grep "Accuracy" | grep -oE '[0-9.]+%' | head -1 || echo "ERROR")
if [[ "$ROUTING" != "ERROR" ]]; then
    ok "routing golden set → accuracy $ROUTING"
else
    fail "routing eval falló"
fi

# ── Preguntas de prueba al agente ─────────────────────────────────────────
if [[ -n "$API_KEY" && "$READY" == "ok" ]]; then
    echo ""
    echo "Preguntas de prueba al agente:"

    declare -a QUERIES=(
        "¿Qué decidió el Consejo del Banco Central en su última reunión?"
        "¿Cuáles son las proyecciones de inflación del BCCh?"
        "¿Qué riesgos externos menciona el Consejo en las minutas?"
        "¿Cuánto vale el USD/CLP actualmente?"
        "Dame el precio del cobre histórico"
    )

    for i in "${!QUERIES[@]}"; do
        Q="${QUERIES[$i]}"
        RESULT=$(curl -sf "$API_URL/v1/chat" \
            -H "X-API-Key: $API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"message\": \"$Q\"}" \
            --max-time 120 \
            | python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get('response',''); print('OK' if len(t)>20 else 'EMPTY')" 2>/dev/null || echo "ERROR")

        if [[ "$RESULT" == "OK" ]]; then
            ok "[$((i+1))] $Q"
        else
            fail "[$((i+1))] $Q → $RESULT"
        fi
    done
fi

# ── Resumen ───────────────────────────────────────────────────────────────
echo ""
echo "=== Resultado: $PASS pasaron, $FAIL fallaron ==="
if [[ "$FAIL" -gt 0 ]]; then
    echo "Ver docs/H100_VALIDATION_CHECKLIST.md para troubleshooting."
    exit 1
else
    echo "Validación exitosa. Generar reporte:"
    echo "  banks-eval all --output docs/H100_VALIDATION_$(date +%Y-%m-%d).md"
fi
