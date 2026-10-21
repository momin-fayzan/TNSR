import { Link } from "react-router-dom";

import { EmptyState } from "../components/ui";

export function NotFound() {
  return (
    <EmptyState title="Page not found" action={<Link className="text-sm text-accent hover:underline" to="/">Back to Studio</Link>}>
      The page you were looking for does not exist.
    </EmptyState>
  );
}
