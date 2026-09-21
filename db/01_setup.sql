-- One-time cluster setup for the welfare DB. Run as the postgres SUPERUSER,
-- passing the role passwords as psql vars (sourced from .env — never stored here):
--
--   set -a; source .env; set +a
--   sudo -u postgres psql -v ON_ERROR_STOP=1 \
--        -v owner_pw="$WELFARE_OWNER_PASSWORD" \
--        -v app_pw="$WELFARE_APP_PASSWORD" \
--        < db/01_setup.sql
--
-- Two roles: welfare_owner writes (DDL + loads), welfare_app only reads.
-- Touches only the new `welfare` DB; the existing `gis` DB is untouched.
-- Re-running errors on the existing roles (harmless — it's one-time).

CREATE ROLE welfare_owner LOGIN PASSWORD :'owner_pw';   -- DDL + all data writes
CREATE ROLE welfare_app   LOGIN PASSWORD :'app_pw';     -- READ-ONLY (the sim/app)

CREATE DATABASE welfare OWNER welfare_owner;

\connect welfare

-- Adds geometry types, ST_* functions and spatial index support to THIS database
-- (the PostGIS package is installed OS-wide, but must be enabled per database).
CREATE EXTENSION IF NOT EXISTS postgis;

-- Raw reference data lives in `poi`; the app's serving tables live in `public`
-- (datasource.py queries them unqualified, i.e. via the default search_path).
CREATE SCHEMA IF NOT EXISTS poi AUTHORIZATION welfare_owner;

-- Postgres 14 lets ANY role create objects in `public` by default; revoke that
-- so welfare_app is genuinely read-only, then hand `public` back to the owner.
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE, CREATE ON SCHEMA public TO welfare_owner;

-- welfare_app: may look into both schemas and SELECT from any table the owner
-- creates in them, now or in future. No INSERT/UPDATE/DELETE, no DDL, ever.
GRANT USAGE ON SCHEMA public, poi TO welfare_app;
ALTER DEFAULT PRIVILEGES FOR ROLE welfare_owner IN SCHEMA public, poi
    GRANT SELECT ON TABLES TO welfare_app;
