#!/bin/sh
set -ae
cd "$(dirname $0)"

source .env
uvicorn api.main:app "$@"
