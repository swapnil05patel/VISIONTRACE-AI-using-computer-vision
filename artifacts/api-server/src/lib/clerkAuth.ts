import { getAuth } from "@clerk/express";
import type { Request, Response } from "express";

function developmentUser(req: Request): string | null {
  if (process.env.NODE_ENV === "production") return null;
  const configured = process.env.VISIONTRACE_DEV_USER_ID;
  if (!configured) return null;
  const header = req.header("x-visiontrace-dev-user");
  if (header && header === configured) return configured;
  if (req.header("x-visiontrace-dev-user") && header !== configured) return null;
  return configured;
}

export function authenticatedUserId(req: Request, res: Response): string | null {
  let clerkUserId: string | null = null;
  if (process.env.CLERK_SECRET_KEY) {
    try {
      clerkUserId = getAuth(req).userId ?? null;
    } catch {
      clerkUserId = null;
    }
  }
  const userId = clerkUserId ?? developmentUser(req);
  if (!userId) {
    res.status(401).json({ error: "Sign in is required to access private footage." });
    return null;
  }
  return userId;
}

export function requireAdmin(req: Request, res: Response): string | null {
  const userId = authenticatedUserId(req, res);
  if (!userId) return null;
  const admins = (process.env.VISIONTRACE_ADMIN_USER_IDS ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  if (admins.length === 0) {
    if (process.env.NODE_ENV === "production") {
      res.status(403).json({ error: "Model management is not configured for any administrator." });
      return null;
    }
    return userId;
  }
  if (!admins.includes(userId)) {
    res.status(403).json({ error: "Administrator access is required for model management." });
    return null;
  }
  return userId;
}
