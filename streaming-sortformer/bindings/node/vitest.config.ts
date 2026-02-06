import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['test/**/*.test.ts'],
    testTimeout: 120000, // Native inference can be slow, especially first load
    hookTimeout: 60000, // Model loading in beforeAll
  },
});
