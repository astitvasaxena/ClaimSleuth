#!/bin/sh
# Start the app. With LITESTREAM_BUCKET set, the SQLite database is restored from the bucket (if a replica
# exists) and replicated to it continuously; otherwise uvicorn runs on its own.
set -e
APP="uvicorn claimsleuth.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"
if [ -n "$LITESTREAM_BUCKET" ]; then
    litestream restore -config /etc/litestream.yml -if-db-not-exists -if-replica-exists "$DATA_DIR/claimsleuth.db"
    exec litestream replicate -config /etc/litestream.yml -exec "$APP"
fi
exec $APP
