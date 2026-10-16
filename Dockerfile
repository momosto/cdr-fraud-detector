# Stage 1: build the dashboard.
FROM node:22-slim AS dashboard
WORKDIR /dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

# Stage 2: generate the CDRs, run detection, serve the API and the dashboard.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY cdrfraud cdrfraud
COPY --from=dashboard /dashboard/dist /app/dashboard
RUN python -m cdrfraud.pipeline --out /app/dashboard/data && python -m cdrfraud.benchmark --seeds 1 2 3 --out /app/dashboard/data/benchmark.json
ENV CDR_DATA_DIR=/app/dashboard/data CDR_DASHBOARD_DIR=/app/dashboard
EXPOSE 8000
CMD ["uvicorn", "cdrfraud.api:app", "--host", "0.0.0.0", "--port", "8000"]
