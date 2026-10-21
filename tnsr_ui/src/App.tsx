import { Route, Routes } from "react-router-dom";

import { Shell } from "./components/Shell";
import { AppProvider } from "./lib/app-context";
import { ExperimentPage } from "./pages/ExperimentPage";
import { ComparePage } from "./pages/ComparePage";
import { LabPage } from "./pages/LabPage";
import { ModelsPage } from "./pages/ModelsPage";
import { NotFound } from "./pages/NotFound";
import { SettingsPage } from "./pages/SettingsPage";
import { StudioPage } from "./pages/StudioPage";

export function App() {
  return (
    <AppProvider>
      <Shell>
        <Routes>
          <Route path="/" element={<StudioPage />} />
          <Route path="/c/:id" element={<StudioPage />} />
          <Route path="/lab" element={<LabPage />} />
          <Route path="/lab/runs/:runId" element={<ExperimentPage />} />
          <Route path="/lab/compare" element={<ComparePage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Shell>
    </AppProvider>
  );
}
