const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  // Window controls
  minimize: ()                      => ipcRenderer.send('win-minimize'),
  maximize: ()                      => ipcRenderer.send('win-maximize'),
  close:    ()                      => ipcRenderer.send('win-close'),
  onWinState: (cb)                  => ipcRenderer.on('win-state', (_e, s) => cb(s)),

  // Dialogs
  openFileDialog: ()                => ipcRenderer.invoke('open-file-dialog'),
  saveFileDialog: (name)            => ipcRenderer.invoke('save-file-dialog', name),
  saveJsonDialog: (name)            => ipcRenderer.invoke('save-json-dialog', name),

  // File I/O
  getFileStats:  (p)                => ipcRenderer.invoke('get-file-stats', p),
  readFileHead:  (p, max)           => ipcRenderer.invoke('read-file-head', p, max),
  readFileChunk: (p, off, sz)       => ipcRenderer.invoke('read-file-chunk', p, off, sz),
  writeJson:     (p, content)       => ipcRenderer.invoke('write-json', p, content),

  // GGUF rebuild
  rebuildGguf:  (inp, out, hdr, dataOff) =>
                                    ipcRenderer.invoke('rebuild-gguf', inp, out, hdr, dataOff),

  // Shell
  showInFolder: (p)                 => ipcRenderer.invoke('show-in-folder', p),

  // Versions
  getVersions: ()                   => ipcRenderer.invoke('get-versions'),
});
