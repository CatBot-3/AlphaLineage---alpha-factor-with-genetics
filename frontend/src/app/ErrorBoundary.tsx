// A small error boundary so a render exception in one view (e.g. a malformed lineage) shows a
// contained message instead of unmounting the whole app to a blank page.

import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // eslint-disable-next-line no-console
    console.error("View crashed:", error, info);
  }

  reset = () => this.setState({ error: null });

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="error-boundary surface-message" data-testid="error-boundary" role="alert">
          <p className="error">Something went wrong rendering this view.</p>
          <p className="hint">{this.state.error.message}</p>
          <button type="button" className="ghost" onClick={this.reset}>
            Dismiss
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
