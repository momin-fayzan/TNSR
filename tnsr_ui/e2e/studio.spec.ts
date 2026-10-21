// Full-stack E2E: a real TNSR backend serving real trained weights (started by scripts/e2e.sh).
import { expect, test } from "@playwright/test";

test("chat streams generated text, stops, persists and shows Lab results", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("status").first()).toContainText("Backend online");

  // 1. New chat, send a message, observe streaming then completion.
  await page.getByRole("button", { name: "New chat" }).click();
  await page.waitForURL(/\/c\//);
  const box = page.getByLabel("Message", { exact: true });
  await box.fill("Hello!");
  await box.press("Enter");
  await expect(page.getByTestId("message-scroller").getByText("Hello!", { exact: true })).toBeVisible();
  const reply = page.getByRole("article", { name: "TNSR response" }).first();
  await expect(reply).toHaveAttribute("data-status", "complete");
  const text = (await reply.textContent()) ?? "";
  expect(text.replace("TNSR", "").trim().length).toBeGreaterThan(0);
  await expect(reply.getByTitle("Generation statistics")).toContainText("tok");
  await expect(page.getByRole("heading", { name: "Hello!" })).toBeVisible(); // auto-title from the first message

  // 2. Observe tokens arriving incrementally, then cancel a long generation.
  await page.getByRole("radio", { name: "Completion" }).click();
  await page.getByRole("button", { name: "Show generation settings" }).click();
  await page.getByLabel("Max new tokens", { exact: true }).fill("4000");
  await page.getByRole("switch", { name: "Ignore end-of-text" }).click();
  await page.waitForTimeout(600); // settings are persisted with a short debounce
  // Record every text length the streaming response passes through, independent of model speed.
  await page.evaluate(() => {
    const w = window as unknown as { __lengths: number[] };
    w.__lengths = [];
    new MutationObserver(() => {
      const el = document.querySelector('article[data-status="streaming"]');
      if (el) w.__lengths.push((el.textContent ?? "").length);
    }).observe(document.body, { subtree: true, childList: true, characterData: true });
  });
  await box.fill("Once upon a time");
  await box.press("Enter");
  await expect(page.locator('article[data-status="streaming"]')).toBeVisible();
  await expect.poll(() => page.evaluate(() => new Set((window as unknown as { __lengths: number[] }).__lengths).size)).toBeGreaterThan(5);
  await page.getByLabel("Stop generation").click();
  const stopped = page.getByRole("article", { name: "TNSR response" }).last();
  await expect(stopped).toHaveAttribute("data-status", "cancelled");
  await expect(stopped.getByText("stopped", { exact: true })).toBeVisible();

  // 3. Reload: the conversation and both responses persist.
  const url = page.url();
  await page.reload();
  await expect(page).toHaveURL(url);
  await expect(page.getByTestId("message-scroller").getByText("Hello!", { exact: true })).toBeVisible();
  await expect(page.getByRole("article", { name: "TNSR response" })).toHaveCount(2);
  await expect(page.locator('article[data-status="cancelled"]')).toHaveCount(1);

  // 4. Rename the conversation from the sidebar.
  await page.getByRole("button", { name: "Rename conversation" }).first().click({ force: true });
  const title = page.getByLabel("Conversation title");
  await title.fill("E2E conversation");
  await title.press("Enter");
  await expect(page.getByRole("heading", { name: "E2E conversation" })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("heading", { name: "E2E conversation" })).toBeVisible();

  // 5. Lab shows the real training run with its loss curve.
  await page.getByRole("link", { name: "Lab" }).click();
  const runLink = page.locator('a[href^="/lab/runs/"]').first();
  await expect(runLink).toBeVisible();
  await runLink.click();
  await expect(page.getByRole("heading", { name: "Loss" })).toBeVisible();
  await expect(page.locator(".recharts-line").first()).toBeVisible();
  await expect(page.getByText("completed").first()).toBeVisible();
});
