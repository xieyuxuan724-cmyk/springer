/** MOCK_CDN=1 verifies HF-like MIME locally; omit it for a real deployment. */
import { chromium } from 'playwright-core';
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
const target = process.env.GAME_URL || 'http://127.0.0.1:17863/';
const browser = await chromium.launch({
  executablePath: process.env.CHROME_BIN || '/usr/bin/google-chrome',
  headless: true,
  args: [
    '--no-sandbox',
    '--enable-unsafe-swiftshader',
    '--use-angle=swiftshader',
    '--disable-dev-shm-usage',
  ],
});
const context = await browser.newContext();
const page = await context.newPage();
const errors = [],
  requests = [];
page.on('pageerror', (error) => errors.push(String(error)));
context.on('request', (request) => requests.push(request.url()));
if (process.env.MOCK_CDN === '1') {
  const mock = async (route) => {
    const url = new URL(route.request().url());
    const path =
      url.hostname === 'cdn.openbeni.test'
        ? url.pathname
        : url.pathname.split('/dist')[1];
    if (!path) return route.continue();
    if (process.env.FAIL_CDN === '1')
      return route.fulfill({
        status: 404,
        headers: { 'access-control-allow-origin': '*' },
        body: 'Missing',
      });
    try {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        headers: { 'access-control-allow-origin': '*' },
        body: await readFile(new URL('../../dist' + path, import.meta.url)),
      });
    } catch {
      await route.fulfill({ status: 404, body: 'Missing' });
    }
  };
  await context.route('https://cdn.openbeni.test/**', mock);
  await context.route('https://huggingface.co/**', mock);
}
const ready = () =>
  page.waitForFunction(
    () =>
      [...document.querySelectorAll('button')].some(
        (b) => b.textContent.includes('Start game') && !b.disabled,
      ),
    null,
    { timeout: 180_000 },
  );
try {
  const start = Date.now();
  await page.goto(target, { waitUntil: 'domcontentloaded', timeout: 90_000 });
  if (process.env.FAIL_CDN === '1') {
    await page
      .getByRole('button', { name: 'Try backup source (may be slower)' })
      .waitFor();
    assert.equal(
      requests.some((url) => url.startsWith(target + 'assets/')),
      false,
      'No automatic origin fallback',
    );
    await page
      .getByRole('button', { name: 'Try backup source (may be slower)' })
      .click();
  }
  await page
    .getByPlaceholder('Choose a name')
    .fill('Loading check', { timeout: 90_000 });
  await page.getByRole('button', { name: 'Enter garage' }).click();
  await ready();
  console.log('Cold ready ms:', Date.now() - start);
  for (const name of ['Toy Harbor', 'Candy Junction', 'Cloud Climb']) {
    await page.getByText(name, { exact: true }).click();
    await ready();
  }
  await page.evaluate(() => {
    for (const name of ['Candy Junction', 'Toy Harbor', 'Cloud Climb']) {
      [...document.querySelectorAll('button')]
        .find((b) => b.textContent.includes(name))
        ?.click();
    }
  });
  await ready();
  const before = requests.length,
    warmStart = Date.now();
  await page.reload({ waitUntil: 'domcontentloaded' });
  await ready();
  const warm = requests
    .slice(before)
    .filter((url) => /\/(runtime|robot|policies|assets)\//.test(url));
  const repeated = warm.filter(
    (url) => !url.endsWith('.mjs') && requests.slice(0, before).includes(url),
  );
  if (process.env.FAIL_CDN !== '1')
    assert.deepEqual(
      repeated,
      [],
      'Warm reload reuses previously downloaded files',
    );
  console.log(
    'Warm ready ms:',
    Date.now() - warmStart,
    'new resource requests:',
    warm,
  );
  await page.getByRole('button', { name: 'Start game', exact: true }).click();
  assert.deepEqual(errors, []);
  console.log(
    'PASS: startup, three courses, rapid selection, warm cache and game entry',
  );
} catch (error) {
  console.error(
    'UI:',
    (await page.locator('body').innerText()).slice(0, 2500),
    'errors:',
    errors,
  );
  throw error;
} finally {
  await browser.close();
}
