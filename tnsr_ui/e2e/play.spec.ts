// Full-stack E2E: real backend, real Stockfish-labeled data, real trained chess weights
// (prepared by scripts/e2e.sh).
import { expect, test } from "@playwright/test";

const square = (sq: string) => `[data-square="${sq}"]`;

test("play TNSR: moves, engine replies, take back, persistence and Lab", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("status").first()).toContainText("Backend online");

  // 1. Start a game as white: the board shows the start position and it is our move.
  await page.getByRole("button", { name: "Play as white" }).click();
  await page.waitForURL(/\/g\//);
  await expect(page.locator(square("e1"))).toHaveAttribute("aria-label", "e1, white king");
  await expect(page.getByText("Your move")).toBeVisible();

  // 2. Play e2-e4 by clicking; TNSR replies with a legal black move.
  await page.locator(square("e2")).click();
  await expect(page.locator(square("e4"))).toHaveAttribute("aria-label", /legal target/);
  await page.locator(square("e4")).click();
  await expect(page.locator(square("e4"))).toHaveAttribute("aria-label", "e4, white pawn");
  await expect(page.getByText(/^TNSR played /)).toBeVisible();
  await expect(page.getByText("Your move")).toBeVisible();
  const details = page.getByRole("complementary", { name: "Game details" });
  await expect(details.getByText("e4", { exact: true }).first()).toBeVisible();

  // An illegal destination does nothing.
  await page.locator(square("d2")).click();
  await page.locator(square("d5")).click();
  await expect(page.locator(square("d2"))).toHaveAttribute("aria-label", "d2, white pawn");

  // 3. Play a second move, then take it back.
  await page.locator(square("g1")).click();
  await page.locator(square("f3")).click();
  await expect(page.locator(square("f3"))).toHaveAttribute("aria-label", "f3, white knight");
  await page.getByRole("button", { name: "Take back" }).click();
  await expect(page.locator(square("g1"))).toHaveAttribute("aria-label", "g1, white knight");
  await expect(page.locator(square("f3"))).toHaveAttribute("aria-label", "f3");

  // 4. Reload: the game persists.
  const url = page.url();
  await page.reload();
  await expect(page).toHaveURL(url);
  await expect(page.locator(square("e4"))).toHaveAttribute("aria-label", "e4, white pawn");

  // 5. Playing black: TNSR moves first.
  await page.getByRole("button", { name: "Play black" }).click();
  await expect(page.getByText(/^TNSR played /)).toBeVisible();
  await expect(page.locator(square("e1"))).toHaveAttribute("aria-label", "e1, white king");

  // 6. Lab shows the chess run with its move-quality chart.
  await page.getByRole("link", { name: "Lab" }).click();
  const runLink = page.locator('a[href^="/lab/runs/"]').first();
  await expect(runLink).toBeVisible();
  await runLink.click();
  await expect(page.getByRole("heading", { name: "Top move = Stockfish" })).toBeVisible();
  await expect(page.locator(".recharts-line").first()).toBeVisible();
});
