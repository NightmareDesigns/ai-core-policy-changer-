const {
  app, BrowserWindow, ipcMain, dialog, shell, nativeTheme
} = require('electron');
const path  = require('path');
const fs    = require('fs');

nativeTheme.themeSource = 'dark';

let mainWindow;

// ─── Window ──────────────────────────────────────────────────────────────────

function createWindow () {
  mainWindow = new BrowserWindow({
    width:          1440,
    height:         900,
    minWidth:       1100,
    minHeight:      680,
    backgroundColor: '#080808',
    frame:          false,
    titleBarStyle:  'hidden',
    webPreferences: {
      preload:          path.join(__dirname, 'preload.js'),
      nodeIntegration:  false,
      contextIsolation: true,
      sandbox:          false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'src', 'index.html'));

  mainWindow.on('closed', () => { mainWindow = null; });

  // Forward maximize/restore state changes
  mainWindow.on('maximize',   () => mainWindow.webContents.send('win-state', 'maximized'));
  mainWindow.on('unmaximize', () => mainWindow.webContents.send('win-state', 'normal'));
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });

// ─── Window controls ─────────────────────────────────────────────────────────

ipcMain.on('win-minimize', () => mainWindow.minimize());
ipcMain.on('win-maximize', () => {
  mainWindow.isMaximized() ? mainWindow.unmaximize() : mainWindow.maximize();
});
ipcMain.on('win-close', () => mainWindow.close());

// ─── File dialogs ─────────────────────────────────────────────────────────────

ipcMain.handle('open-file-dialog', async () => {
  return dialog.showOpenDialog(mainWindow, {
    title:      'Select GGUF File',
    properties: ['openFile'],
    filters: [
      { name: 'GGUF Models', extensions: ['gguf'] },
      { name: 'All Files',   extensions: ['*'] },
    ],
  });
});

ipcMain.handle('save-file-dialog', async (_e, defaultName) => {
  return dialog.showSaveDialog(mainWindow, {
    title:       'Save Rebuilt GGUF',
    defaultPath: defaultName || 'output.gguf',
    filters: [
      { name: 'GGUF Models', extensions: ['gguf'] },
      { name: 'All Files',   extensions: ['*'] },
    ],
  });
});

ipcMain.handle('save-json-dialog', async (_e, defaultName) => {
  return dialog.showSaveDialog(mainWindow, {
    title:       'Export Metadata as JSON',
    defaultPath: defaultName || 'metadata.json',
    filters: [{ name: 'JSON', extensions: ['json'] }],
  });
});

// ─── File I/O ────────────────────────────────────────────────────────────────

ipcMain.handle('get-file-stats', async (_e, filePath) => {
  try {
    const s = fs.statSync(filePath);
    return { success: true, size: s.size, mtime: s.mtime.toISOString(), birthtime: s.birthtime.toISOString() };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Read the first `maxBytes` bytes of a file (for header parsing)
ipcMain.handle('read-file-head', async (_e, filePath, maxBytes) => {
  try {
    const fd    = fs.openSync(filePath, 'r');
    const buf   = Buffer.alloc(maxBytes);
    const read  = fs.readSync(fd, buf, 0, maxBytes, 0);
    fs.closeSync(fd);
    const slice = buf.slice(0, read);
    // Transfer as plain array so IPC can handle it
    return { success: true, data: Array.from(slice), bytesRead: read };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Read an arbitrary chunk (for hex viewer)
ipcMain.handle('read-file-chunk', async (_e, filePath, offset, size) => {
  try {
    const fd   = fs.openSync(filePath, 'r');
    const buf  = Buffer.alloc(size);
    const read = fs.readSync(fd, buf, 0, size, offset);
    fs.closeSync(fd);
    return { success: true, data: Array.from(buf.slice(0, read)), bytesRead: read };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Write JSON to disk
ipcMain.handle('write-json', async (_e, filePath, content) => {
  try {
    fs.writeFileSync(filePath, content, 'utf8');
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// ─── GGUF streaming rebuild ───────────────────────────────────────────────────
// Takes a pre-built header (as number array) and streams the tensor-data
// section from the original file into a new output file.
ipcMain.handle('rebuild-gguf', async (_e, inputPath, outputPath, headerArray, originalDataOffset) => {
  try {
    const headerBuf = Buffer.from(headerArray);
    const alignment = 32; // default GGUF alignment

    // Align new header to alignment boundary
    const paddedLen  = Math.ceil(headerBuf.length / alignment) * alignment;
    const padding    = Buffer.alloc(paddedLen - headerBuf.length, 0);

    const outFd = fs.openSync(outputPath, 'w');
    fs.writeSync(outFd, headerBuf);
    fs.writeSync(outFd, padding);

    // Stream tensor data
    const inFd     = fs.openSync(inputPath, 'r');
    const CHUNK    = 4 * 1024 * 1024; // 4 MiB chunks
    const chunk    = Buffer.alloc(CHUNK);
    let   srcOff   = originalDataOffset;
    const fileSize = fs.statSync(inputPath).size;

    while (srcOff < fileSize) {
      const toRead = Math.min(CHUNK, fileSize - srcOff);
      const read   = fs.readSync(inFd, chunk, 0, toRead, srcOff);
      if (read === 0) break;
      fs.writeSync(outFd, chunk, 0, read);
      srcOff += read;
    }

    fs.closeSync(inFd);
    fs.closeSync(outFd);
    return { success: true, bytesWritten: paddedLen + (fileSize - originalDataOffset) };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Open path in system file manager
ipcMain.handle('show-in-folder', async (_e, filePath) => {
  shell.showItemInFolder(filePath);
  return { success: true };
});

// Runtime info
ipcMain.handle('get-versions', async () => ({
  node:     process.versions.node,
  electron: process.versions.electron,
  chrome:   process.versions.chrome,
  platform: process.platform,
  arch:     process.arch,
}));
