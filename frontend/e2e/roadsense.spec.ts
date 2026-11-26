import {expect, test} from '@playwright/test'
import {fileURLToPath} from 'node:url'

const photo = fileURLToPath(new URL('./fixture.png', import.meta.url))

test('upload, inference, comparison, benchmark and history use the real fixture API', async ({page}, info) => {
  await page.goto('/')
  await expect(page.getByText('3 models available')).toBeVisible()
  const columns = await page.locator('.analyze-grid').evaluate(grid =>
    getComputedStyle(grid).gridTemplateColumns.split(' ').length)
  expect(columns).toBe(info.project.name === 'desktop' ? 2 : 1)
  await expect(page.getByRole('button',{name:'Analyze photo'})).toBeDisabled()
  await page.getByLabel('Choose a road photo').setInputFiles(photo)
  await expect(page.getByAltText('Selected road surface')).toBeVisible()
  await page.route('**/predict/resnet50', async route => {
    await new Promise(resolve => setTimeout(resolve, 350))
    await route.continue()
  })
  await page.getByRole('button',{name:'Analyze photo'}).click()
  await expect(page.getByText('Analyzing road surface…')).toBeVisible()
  await expect(page.getByRole('heading',{name:'Poor road quality'})).toBeVisible()
  await expect(page.getByText('server inference')).toBeVisible()
  await page.getByRole('link',{name:'Compare this photo'}).click()
  await page.getByRole('button',{name:'Run comparison'}).click()
  await expect(page.getByRole('heading',{name:'MobileNetV2'})).toBeVisible()
  await expect(page.getByRole('heading',{name:'ResNet50'})).toBeVisible()
  await expect(page.getByRole('heading',{name:'EfficientNetB0'})).toBeVisible()
  await expect(page.getByText('models assigned different')).toBeVisible()
  const nav = page.getByRole('navigation',{name:'Main navigation'})
  await nav.getByRole('link',{name:'Benchmark'}).click()
  await expect(page.getByRole('heading',{name:'Confusion matrix'})).toBeVisible()
  await expect(page.getByText('version isolated-test-fixture')).toBeVisible()
  await nav.getByRole('link',{name:'History'}).click()
  await expect(page.getByRole('heading',{name:'Scan history'})).toBeVisible()
  await expect(page.getByRole('listitem').filter({hasText:'ResNet50'})).toBeVisible()
  await page.reload()
  await expect(page.getByRole('heading',{name:'Scan history'})).toBeVisible()
  await expect(page.getByRole('listitem').filter({hasText:'ResNet50'})).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
    await page.evaluate(() => window.innerWidth))
})

test('invalid upload, API failure and retry provide actionable feedback', async ({page}) => {
  await page.goto('/')
  await expect(page.getByText('3 models available')).toBeVisible()
  await page.getByLabel('Choose a road photo').setInputFiles({
    name:'not-a-road.txt', mimeType:'text/plain', buffer:Buffer.from('text'),
  })
  await expect(page.getByRole('alert')).toHaveText('Choose a JPEG, PNG, or WebP image.')
  await page.getByLabel('Choose a road photo').setInputFiles(photo)
  await page.route('**/predict/resnet50', route=>route.abort('failed'))
  await page.getByRole('button',{name:'Analyze photo'}).click()
  await expect(page.getByRole('alert')).toContainText('Could not connect')
  await page.unroute('**/predict/resnet50')
  await page.getByRole('button',{name:'Retry'}).click()
  await expect(page.getByRole('heading',{name:'Poor road quality'})).toBeVisible()
})

test('a valid image rejected as non-road prompts for a replacement', async ({page}) => {
  await page.goto('/')
  await expect(page.getByText('3 models available')).toBeVisible()
  await page.getByLabel('Choose a road photo').setInputFiles({
    name:'non-road.png', mimeType:'image/png',
    buffer:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXWQAAAAASUVORK5CYII=', 'base64'),
  })
  await page.getByRole('button',{name:'Analyze photo'}).click()
  await expect(page.getByRole('alert')).toContainText('Invalid road photo.')
  await expect(page.getByRole('button',{name:'Retry'})).toHaveCount(0)
  await page.getByRole('link',{name:'Choose another photo'}).click()
  await page.getByLabel('Choose a road photo').setInputFiles(photo)
  await page.getByRole('button',{name:'Analyze photo'}).click()
  await expect(page.getByRole('heading',{name:'Poor road quality'})).toBeVisible()
})

test('camera capture yields a preview', async ({page},info) => {
  test.skip(info.project.name !== 'desktop', 'Fake Chromium camera enabled in desktop project')
  await page.goto('/')
  await page.getByRole('link',{name:'Open capture workspace'}).click()
  await expect(page.getByRole('heading',{name:'Capture road photo'})).toBeVisible()
  await page.getByRole('button',{name:'Take photo'}).click()
  const video = page.getByLabel('Live road camera')
  await expect(video).toBeVisible()
  await expect.poll(() => video.evaluate((node: HTMLVideoElement) => node.videoWidth)).toBeGreaterThan(0)
  await page.getByRole('button',{name:'Capture photo'}).click()
  await expect(page.getByAltText('Selected road surface')).toBeVisible()
  await expect(page.getByRole('button',{name:'Retake photo'})).toBeVisible()
})
