FROM node:24.11.1-bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY web/package.json web/package-lock.json ./web/
RUN cd web && npm ci --ignore-scripts
COPY . .
ARG VITE_ASSET_BASE_URL=https://huggingface.co/spaces/mondo-robotics/open-beni/resolve/9962938a66f203a3715cfd20230f61eb11c42141/dist
RUN cd web && npm run build

FROM node:24.11.1-bookworm-slim
WORKDIR /app/web
COPY --from=build /app/web/package.json ./
COPY --from=build /app/web/node_modules/ws ./node_modules/ws
COPY --from=build /app/web/server ./server
COPY --from=build /app/web/shared ./shared
COPY --from=build /app/web/game ./game
COPY --from=build /app/web/.generated/loaders.json ./.generated/loaders.json
COPY --from=build /app/web/dist ./dist
ENV NODE_ENV=production HOST=0.0.0.0 PORT=7860 MAX_ROOMS=10
# Set FRONTEND_ASSET_BASE_URL at runtime only for a matching published build.
USER node
EXPOSE 7860
CMD ["node", "server/index.mjs"]
