#!/bin/bash
set -a
source .env
set +a

litellm --config ./config.yaml

