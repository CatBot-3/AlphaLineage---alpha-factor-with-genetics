import type { AppMode } from "../api/types";

export function getAppMode(): AppMode {
  return import.meta.env.VITE_DATA_SOURCE === "app" ? "app" : "demo";
}

export function modeLabel(mode: AppMode): string {
  return mode === "app" ? "Local Backend" : "Static Demo";
}
