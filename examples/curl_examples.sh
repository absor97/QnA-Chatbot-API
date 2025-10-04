#!/bin/bash

# RAG Question Answering API - cURL Examples
# This script demonstrates how to interact with the API using cURL

BASE_URL="http://localhost:8000"

echo "========================================="
echo "RAG Question Answering API - cURL Examples"
echo "========================================="
echo ""

# 1. Health Check
echo "1️⃣ Health Check"
echo "-----------------------------------"
curl -X GET "$BASE_URL/health" | jq
echo ""
echo ""

# 2. Ask about return policy
echo "2️⃣ Ask: What is the return policy?"
echo "-----------------------------------"
curl -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}' | jq
echo ""
echo ""

# 3. Ask about shipping
echo "3️⃣ Ask: What are the shipping options?"
echo "-----------------------------------"
curl -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the shipping options?"}' | jq
echo ""
echo ""

# 4. Ask about products
echo "4️⃣ Ask: Tell me about the Smart Home Hub Pro"
echo "-----------------------------------"
curl -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about the Smart Home Hub Pro"}' | jq
echo ""
echo ""

# 5. Ask about customer support
echo "5️⃣ Ask: How can I contact customer support?"
echo "-----------------------------------"
curl -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How can I contact customer support?"}' | jq
echo ""
echo ""

# 6. Ask about warranty
echo "6️⃣ Ask: What is the warranty policy?"
echo "-----------------------------------"
curl -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the warranty policy?"}' | jq
echo ""
echo ""

echo "========================================="
echo "✅ Examples completed!"
echo "========================================="
