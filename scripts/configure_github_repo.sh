#!/usr/bin/env bash
# Configure GitHub repository description and topics for Conflux.
# Requires `gh` authenticated with repository administration permissions.
set -euo pipefail

REPO="${1:-lolllll212/Conflux}"

gh repo edit "$REPO" \
  --description "Deterministic, lock-free CRDT shared state protocol for concurrent AI agents, with drop-in adapters for LangGraph, CrewAI, and AutoGen." \
  --homepage "https://pypi.org/project/conflux-state/" \
  --add-topic multi-agent \
  --add-topic crdt \
  --add-topic distributed-systems \
  --add-topic ai-agents \
  --add-topic langgraph \
  --add-topic crewai \
  --add-topic autogen \
  --add-topic shared-state \
  --add-topic python

echo "Successfully updated description and topics for $REPO:"
gh api "repos/$REPO" --jq '{full_name, description, homepage, topics}'
