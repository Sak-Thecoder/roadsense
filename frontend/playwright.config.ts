import {defineConfig, devices} from '@playwright/test'

export default defineConfig({
  testDir:'./e2e',
  timeout:30_000,
  use:{baseURL:'http://127.0.0.1:5179'},
  projects:[
    {name:'desktop', use:{...devices['Desktop Chrome'], viewport:{width:1024,height:768},
      permissions:['camera'], launchOptions:{args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']}}},
    {name:'mobile', use:{...devices['Pixel 7']}},
    {name:'portrait tablet', use:{...devices['Desktop Chrome'], viewport:{width:768,height:1024}}},
  ],
  webServer:[
    {command:'cd .. && TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -m backend.tests.fixture_server',
      url:'http://127.0.0.1:8129/health', timeout:90_000},
    {command:'ROADSENSE_DEV_API=http://127.0.0.1:8129 npm run dev -- --host 127.0.0.1 --port 5179 --strictPort',
      url:'http://127.0.0.1:5179/', timeout:30_000},
  ],
})
