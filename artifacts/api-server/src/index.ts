import app from "./app";
import { logger } from "./lib/logger";
import { startJobWorker } from "./lib/jobWorker";
import { startMlSidecar } from "./lib/mlSidecar";

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

const mlSidecar = startMlSidecar();
const jobWorker = startJobWorker();
const server = app.listen(port, (err) => {
  if (err) {
    logger.error({ err }, "Error listening on port");
    process.exit(1);
  }

  logger.info({ port }, "Server listening");
});

const shutdown = () => {
  clearInterval(jobWorker);
  mlSidecar?.kill("SIGTERM");
  server.close(() => process.exit(0));
};

process.once("SIGTERM", shutdown);
process.once("SIGINT", shutdown);
