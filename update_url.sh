#!/bin/bash

BACKEND_SET="https://etbaly.yussefrostom.me/api/v1/admin/ai/set-image-to-3d-url"
BACKEND_GET="https://etbaly.yussefrostom.me/api/v1/admin/ai/image-to-3d-url"

echo "Current URL in backend:"
curl -s -X GET $BACKEND_GET
echo ""
echo "─────────────────────────────────────"

echo "Paste your new Lightning.ai URL:"
read NEW_URL

if [ -z "$NEW_URL" ]; then
    echo "No change made."
    exit 0
fi

echo "Updating..."
curl -s -X POST $BACKEND_SET \
  -H "Content-Type: application/json" \
  -d "{\"url\": \"$NEW_URL\"}"

echo ""
echo "─────────────────────────────────────"
echo "Verifying..."
curl -s -X GET $BACKEND_GET
echo ""
echo "✅ Done!"
