import { GetVisionRuntimeResponse } from "@workspace/api-zod";

const runtimeUrl =
  process.env.VISIONTRACE_ML_URL ?? "http://127.0.0.1:18777";

const unavailableRuntime = {
  status: "unavailable" as const,
  detector: "unavailable",
  reid: "unavailable",
  vectorSearch: "unavailable",
  device: "unavailable" as const,
  verifiedAt: null,
  limitations: [
    "The VisionTrace ML sidecar could not be reached.",
    "No inference or search results are available until the configured models pass verification.",
  ],
};

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  function mlHeaders(init?: ConstructorParameters<typeof Headers>[0]): Headers {
                                                                                                                                                                                const headers = new Headers(init);
  const secret = process.env.VISIONTRACE_ML_SHARED_SECRET;
  if (secret) headers.set("x-visiontrace-ml-secret", secret);
  return headers;
}

export async function getVisionRuntime() {
  try {
    const response = await fetch(`${runtimeUrl}/healthz`, {
      signal: AbortSignal.timeout(1500),
    });
    if (!response.ok) return GetVisionRuntimeResponse.parse(unavailableRuntime);
    const data = (await response.json()) as {
      runtime?: unknown;
    };
    return GetVisionRuntimeResponse.parse(data.runtime ?? unavailableRuntime);
  } catch {
    return GetVisionRuntimeResponse.parse(unavailableRuntime);
  }
}

export async function callMl<T>(
  path: string,
  init: RequestInit,
  timeoutMs = 30_000,
): Promise<{ ok: true; data: T } | { ok: false; status: number; error: string }> {
  try {
    const response = await fetch(`${runtimeUrl}${path}`, {
      ...init,
      headers: mlHeaders(init.headers),
      signal: AbortSignal.timeout(timeoutMs),
    });
    const payload = (await response.json()) as {
      detail?: unknown;
      error?: unknown;
    };
    if (!response.ok) {
      const detail =
        typeof payload.detail === "string"
          ? payload.detail
          : typeof payload.error === "string"
            ? payload.error
            : "Computer-vision request failed.";
      return { ok: false, status: response.status, error: detail };
    }
    return { ok: true, data: payload as T };
  } catch {
    return {
      ok: false,
      status: 503,
      error: "The VisionTrace ML sidecar is not reachable.",
    };
  }
}
