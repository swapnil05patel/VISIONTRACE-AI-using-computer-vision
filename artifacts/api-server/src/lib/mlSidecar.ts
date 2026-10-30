import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { logger } from "./logger";

export function startMlSidecar(): ChildProcess | null {
  if (process.env.VISIONTRACE_ML_URL) {
    logger.info("Using configured external VisionTrace ML runtime");
    return null;
  }
  const candidates = [
    resolve(process.cwd(), "ml"),
    resolve(process.cwd(), "artifacts/api-server/ml"),
  ];
  const appDirectory = candidates.find((path) => existsSync(resolve(path, "app.py")));
  if (!appDirectory) {
    logger.error("VisionTrace ML sidecar source was not found");
    return null;
  }
  const command = process.env.PYTHON_EXECUTABLE ?? "python3.12";
  const child = spawn(
    command,
    [
      "-m",
      "uvicorn",
      "app:app",
      "--app-dir",
      appDirectory,
      "--host",
      "127.0.0.1",
      "--port",
      process.env.VISIONTRACE_ML_PORT ?? "18777",
      "--workers",
      "1",
    ],
    {
      stdio: ["ignore", "ignore", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    },
  );
  child.stderr?.on("data", (chunk: Buffer) => {
    const line = chunk.toString().trim();
    if (line) logger.info({ source: "visiontrace-ml", line });
  });
  child.on("error", (err) => {
    logger.error({ err }, "Could not start VisionTrace ML sidecar");
  });
  child.on("exit", (code, signal) => {
    if (code !== 0 && code !== null) {
      logger.error({ code, signal }, "VisionTrace ML sidecar exited unexpectedly");
    }
  });
  logger.info({ command, appDirectory }, "Starting VisionTrace ML sidecar");
  return child;
}
