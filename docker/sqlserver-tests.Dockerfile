# The image the repository tests run in against SQL Server: Python, uv and Microsoft's ODBC
# Driver 18, which a developer's machine usually lacks. `uv run poe test-sqlserver` uses it.
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl gnupg \
    && curl --fail --silent --show-error --location https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor --output /usr/share/keyrings/microsoft-prod.gpg \
    && curl --fail --silent --show-error --location \
        https://packages.microsoft.com/config/debian/12/prod.list \
        --output /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install --yes --no-install-recommends msodbcsql18 unixodbc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv

# The repository is mounted at /repo; its .venv belongs to the host, so the image keeps its own.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /repo
