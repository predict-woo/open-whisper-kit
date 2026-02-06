import { createRequire } from 'module';

const require = createRequire(import.meta.url);

let cachedBinding: any = null;

/**
 * Get the native binding for the current platform
 * Detects platform and architecture, loads the appropriate platform-specific package
 * @returns The native module binding
 * @throws Error if platform is not supported or binding cannot be loaded
 */
export function getBinding(): any {
  if (cachedBinding) return cachedBinding;

  const platform = process.platform;
  const arch = process.arch;

  let packageName: string;

  if (platform === 'darwin' && arch === 'arm64') {
    packageName = '@streaming-sortformer-node/darwin-arm64';
  } else if (platform === 'darwin' && arch === 'x64') {
    packageName = '@streaming-sortformer-node/darwin-x64';
  } else {
    throw new Error(
      `Unsupported platform: ${platform}-${arch}. ` +
      `streaming-sortformer-node currently supports: darwin-arm64, darwin-x64`
    );
  }

  try {
    cachedBinding = require(packageName);
    return cachedBinding;
  } catch (e) {
    throw new Error(
      `Failed to load native binding from ${packageName}. ` +
      `Make sure the package is installed: npm install ${packageName}`
    );
  }
}
