// Vitest/jsdom setup. React Flow needs a few browser APIs jsdom does not implement;
// mocking them lets the components mount so the DOM assertions work.

import "@testing-library/jest-dom/vitest";

class ResizeObserverMock {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}
globalThis.ResizeObserver = ResizeObserverMock as unknown as typeof ResizeObserver;

class DOMMatrixMock {
  m22 = 1;
  constructor(_transform?: string) {}
}
globalThis.DOMMatrixReadOnly = DOMMatrixMock as unknown as typeof DOMMatrixReadOnly;
globalThis.DOMMatrix = DOMMatrixMock as unknown as typeof DOMMatrix;

if (!globalThis.matchMedia) {
  globalThis.matchMedia = ((): MediaQueryList =>
    ({
      matches: false,
      addEventListener() {},
      removeEventListener() {},
    }) as unknown as MediaQueryList) as typeof matchMedia;
}

Object.defineProperty(HTMLElement.prototype, "offsetWidth", { configurable: true, value: 800 });
Object.defineProperty(HTMLElement.prototype, "offsetHeight", { configurable: true, value: 600 });
