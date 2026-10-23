import clsx from "clsx";
import { Check, Plus, Pencil, Search, Trash2, X } from "lucide-react";
import { useState, type KeyboardEvent, type ReactNode } from "react";

import { Button, IconButton, Spinner } from "./ui";

export interface SidebarItem {
  id: string;
  title: string;
  meta?: ReactNode;
}

interface Props {
  items: SidebarItem[];
  label?: string;
  noun?: string;
  createLabel?: string;
  activeId: string | null;
  loading: boolean;
  error: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onRename: (id: string, title: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  creating?: boolean;
}

export function ItemSidebar({
  items,
  label = "Games",
  noun = "game",
  createLabel = "New game",
  activeId,
  loading,
  error,
  onSelect,
  onCreate,
  onRename,
  onDelete,
  creating,
}: Props) {
  const [query, setQuery] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const filtered = items.filter((c) => c.title.toLowerCase().includes(query.trim().toLowerCase()));

  const commitRename = async (id: string) => {
    const title = draft.trim();
    setEditing(null);
    if (title && title !== items.find((c) => c.id === id)?.title) await onRename(id, title);
  };

  const onEditKey = (e: KeyboardEvent<HTMLInputElement>, id: string) => {
    if (e.key === "Enter") commitRename(id);
    if (e.key === "Escape") setEditing(null);
  };

  return (
    <aside aria-label={label} className="flex h-full w-full flex-col border-r border-border bg-surface">
      <div className="space-y-2 p-3">
        <Button variant="primary" className="w-full" onClick={onCreate} loading={creating}>
          <Plus className="size-4" aria-hidden /> {createLabel}
        </Button>
        <div className="relative">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-faint" aria-hidden />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={`Search ${noun}s`}
            aria-label={`Search ${noun}s`}
            className="h-8 w-full rounded-md border border-border bg-surface-2 pr-2 pl-8 text-sm placeholder:text-faint focus:border-accent focus:outline-none"
          />
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {loading && items.length === 0 && (
          <div className="flex justify-center py-6">
            <Spinner />
          </div>
        )}
        {error && <p className="px-2 py-3 text-xs text-danger">{error}</p>}
        {!loading && !error && items.length === 0 && (
          <p className="px-2 py-6 text-center text-xs text-faint">No {noun}s yet.</p>
        )}
        {items.length > 0 && filtered.length === 0 && (
          <p className="px-2 py-6 text-center text-xs text-faint">No matches.</p>
        )}
        <ul className="space-y-0.5">
          {filtered.map((c) => (
            <li key={c.id}>
              {editing === c.id ? (
                <div className="flex items-center gap-1 rounded-md bg-surface-2 px-1.5 py-1">
                  <input
                    autoFocus
                    value={draft}
                    maxLength={200}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => onEditKey(e, c.id)}
                    onBlur={() => commitRename(c.id)}
                    aria-label={`${noun[0].toUpperCase()}${noun.slice(1)} title`}
                    className="h-7 min-w-0 flex-1 rounded border border-accent bg-surface px-2 text-sm focus:outline-none"
                  />
                  <IconButton label="Save title" onMouseDown={(e) => e.preventDefault()} onClick={() => commitRename(c.id)}>
                    <Check className="size-3.5" />
                  </IconButton>
                </div>
              ) : confirmDelete === c.id ? (
                <div className="flex items-center justify-between gap-2 rounded-md border border-danger/30 bg-danger/8 px-2 py-1.5 text-xs">
                  <span className="text-danger">Delete this {noun}?</span>
                  <div className="flex gap-1">
                    <IconButton
                      label="Confirm delete"
                      className="text-danger hover:text-danger"
                      onClick={async () => {
                        setConfirmDelete(null);
                        await onDelete(c.id);
                      }}
                    >
                      <Check className="size-3.5" />
                    </IconButton>
                    <IconButton label="Cancel delete" onClick={() => setConfirmDelete(null)}>
                      <X className="size-3.5" />
                    </IconButton>
                  </div>
                </div>
              ) : (
                <div
                  className={clsx(
                    "group flex items-center rounded-md transition-colors",
                    c.id === activeId ? "bg-surface-3" : "hover:bg-surface-2",
                  )}
                >
                  <button
                    onClick={() => onSelect(c.id)}
                    aria-current={c.id === activeId ? "page" : undefined}
                    className="min-w-0 flex-1 px-2.5 py-2 text-left"
                  >
                    <div className={clsx("truncate text-sm", c.id === activeId ? "text-fg" : "text-fg/85")}>{c.title}</div>
                    {c.meta && <div className="mt-0.5 flex gap-1.5 text-[11px] text-faint">{c.meta}</div>}
                  </button>
                  <div className="flex pr-1 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
                    <IconButton
                      label={`Rename ${noun}`}
                      onClick={() => {
                        setDraft(c.title);
                        setEditing(c.id);
                      }}
                    >
                      <Pencil className="size-3.5" />
                    </IconButton>
                    <IconButton label={`Delete ${noun}`} onClick={() => setConfirmDelete(c.id)}>
                      <Trash2 className="size-3.5" />
                    </IconButton>
                  </div>
                </div>
              )}
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
