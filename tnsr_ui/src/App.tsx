import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";

import { Shell } from "./components/Shell";
import { Spinner } from "./components/ui";
import { AppProvider } from "./lib/app-context";
import { NotFound } from "./pages/NotFound";
import { PlayPage } from "./pages/PlayPage";

// Lab pages pull in the charting library; load them on demand so the board opens fast.
const LabPage = lazy(() => import("./pages/LabPage").then((m) => ({ default: m.LabPage })));
const ExperimentPage = lazy(() => import("./pages/ExperimentPage").then((m) => ({ default: m.ExperimentPage })));
const ComparePage = lazy(() => import("./pages/ComparePage").then((m) => ({ default: m.ComparePage })));
const ModelsPage = lazy(() => import("./pages/ModelsPage").then((m) => ({ default: m.ModelsPage })));
const SettingsPage = lazy(() => import("./pages/SettingsPage").then((m) => ({ default: m.SettingsPage })));

function PageFallback() {
  return (
    <div className="flex flex-1 items-center justify-center">
      <Spinner />
    </div>
  );
}

export function App() {
  return (
    <AppProvider>
      <Shell>
        <Suspense fallback={<PageFallback />}>
          <Routes>
            <Route path="/" element={<PlayPage />} />
            <Route path="/g/:id" element={<PlayPage />} />
            <Route path="/lab" element={<LabPage />} />
            <Route path="/lab/runs/:runId" element={<ExperimentPage />} />
            <Route path="/lab/compare" element={<ComparePage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </Shell>
    </AppProvider>
  );
}
