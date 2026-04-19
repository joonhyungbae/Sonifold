import type { MappingStrategy } from "./spectralMapping";

export interface AppBarProps {
  meshLoading: boolean;
  meshError: string | null;
  micOn: boolean;
  micError: string | null;
  meshName: string;
  setMeshName: (v: string) => void;
  meshOptions: { id: string; label: string }[];
  audioStimulus: string;
  setAudioStimulus: (v: string) => void;
  audioStimuli: readonly string[];
  setMicOn: (v: boolean) => void;
  strategy: MappingStrategy;
  setStrategy: (v: MappingStrategy) => void;
  contourLevels: number;
  setContourLevels: (v: number) => void;
  /** When stimulus is "upload", called when user selects or clears a file. */
  onUploadFileChange?: (file: File | null) => void;
  /** When stimulus is "upload", the name of the selected file (for display). */
  uploadedFileName?: string | null;
}

export function AppBarDesktop({
  meshLoading,
  meshError,
  micOn,
  micError,
  meshName,
  setMeshName,
  meshOptions,
  audioStimulus,
  setAudioStimulus,
  audioStimuli,
  setMicOn,
  strategy,
  setStrategy,
  contourLevels,
  setContourLevels,
  onUploadFileChange,
  uploadedFileName,
}: AppBarProps) {
  return (
    <header className="panel panel--top appbar appbar--desktop" role="toolbar" aria-label="Controls">
      {(meshLoading || meshError || (micOn && micError)) && (
        <div className="appbar-messages" role="status" aria-live="polite">
          {meshLoading && <span className="panel-message panel-message--loading">Loading…</span>}
          {meshError && <span className="panel-message panel-message--error" role="alert">{meshError}</span>}
          {micOn && micError && <span className="panel-message panel-message--error" role="alert">{micError}</span>}
        </div>
      )}
      <div className="appbar-item">
        <label htmlFor="mesh-select-d">Mesh</label>
        <select
          id="mesh-select-d"
          className="select"
          value={meshName}
          onChange={(e) => setMeshName(e.target.value)}
          aria-label="Select mesh"
        >
          {meshOptions.map((m) => (
            <option key={m.id} value={m.id}>{m.label}</option>
          ))}
        </select>
      </div>
      <div className="appbar-item appbar-item--stimulus">
        <label htmlFor="stimulus-select-d">Stimulus</label>
        <select
          id="stimulus-select-d"
          className="select"
          value={audioStimulus === "mic" || audioStimulus === "upload" ? "A4" : audioStimulus}
          onChange={(e) => setAudioStimulus(e.target.value)}
          aria-label="Select audio stimulus"
        >
          {audioStimuli.filter((a) => a !== "mic" && a !== "upload").map((a) => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>
      </div>
      <div className="appbar-item">
        <label htmlFor="mapping-select-d">Mapping</label>
        <select
          id="mapping-select-d"
          className="select"
          value={strategy}
          onChange={(e) => setStrategy(e.target.value as MappingStrategy)}
          aria-label="Select mapping strategy"
        >
          <option value="direct">direct</option>
          <option value="mel">mel</option>
          <option value="energy">energy</option>
        </select>
      </div>
      <div className="appbar-item">
        <label htmlFor="contour-select-d">Contours</label>
        <select
          id="contour-select-d"
          className="select"
          value={contourLevels}
          onChange={(e) => setContourLevels(Number(e.target.value))}
          aria-label="Contour levels"
        >
          {[0, 3, 5, 7, 9, 12, 16].map((n) => (
            <option key={n} value={n}>{n === 0 ? "None" : `${n} levels`}</option>
          ))}
        </select>
      </div>
      <div className="appbar-right">
        <button
          type="button"
          className={"appbar-btn" + (audioStimulus === "mic" ? " appbar-btn--active" : "")}
          onClick={() => setAudioStimulus("mic")}
          aria-pressed={audioStimulus === "mic"}
          aria-label="Live microphone"
        >
          Live Mic
        </button>
        {audioStimulus === "mic" && (
          <label className="checkbox-wrap checkbox-wrap--inline">
            <input type="checkbox" checked={micOn} onChange={(e) => setMicOn(e.target.checked)} aria-label="Use microphone" />
            <span className="checkbox-wrap__label">On</span>
          </label>
        )}
        <button
          type="button"
          className={"appbar-btn" + (audioStimulus === "upload" ? " appbar-btn--active" : "")}
          onClick={() => setAudioStimulus("upload")}
          aria-pressed={audioStimulus === "upload"}
          aria-label="Upload audio file"
        >
          Upload
        </button>
        {audioStimulus === "upload" && onUploadFileChange && (
          <label className="appbar-upload appbar-upload--trigger">
            <input
              type="file"
              accept="audio/*"
              className="appbar-upload__input"
              onChange={(e) => onUploadFileChange(e.target.files?.[0] ?? null)}
              aria-label="Choose audio file"
            />
            <span className="appbar-upload__text">
              {uploadedFileName ? (uploadedFileName.length > 18 ? uploadedFileName.slice(0, 15) + "…" : uploadedFileName) : "Choose file"}
            </span>
          </label>
        )}
      </div>
    </header>
  );
}

export function AppBarMobile({
  meshLoading,
  meshError,
  micOn,
  micError,
  meshName,
  setMeshName,
  meshOptions,
  audioStimulus,
  setAudioStimulus,
  audioStimuli,
  setMicOn,
  strategy,
  setStrategy,
  contourLevels,
  setContourLevels,
  onUploadFileChange,
  uploadedFileName,
}: AppBarProps) {
  return (
    <header className="panel panel--top appbar appbar--mobile" role="toolbar" aria-label="Controls">
      {(meshLoading || meshError || (micOn && micError)) && (
        <div className="appbar-messages appbar-messages--mobile" role="status" aria-live="polite">
          {meshLoading && <span className="panel-message panel-message--loading">Loading…</span>}
          {meshError && <span className="panel-message panel-message--error" role="alert">{meshError}</span>}
          {micOn && micError && <span className="panel-message panel-message--error" role="alert">{micError}</span>}
        </div>
      )}
      <div className="appbar-grid appbar-grid--mobile">
        <div className="appbar-item appbar-item--mobile">
          <label htmlFor="mesh-select-m">Mesh</label>
          <select
            id="mesh-select-m"
            className="select"
            value={meshName}
            onChange={(e) => setMeshName(e.target.value)}
            aria-label="Select mesh"
          >
            {meshOptions.map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        </div>
        <div className="appbar-item appbar-item--mobile">
          <label htmlFor="stimulus-select-m">Stimulus</label>
          <select
            id="stimulus-select-m"
            className="select"
            value={audioStimulus === "mic" || audioStimulus === "upload" ? "A4" : audioStimulus}
            onChange={(e) => setAudioStimulus(e.target.value)}
            aria-label="Select audio stimulus"
          >
            {audioStimuli.filter((a) => a !== "mic" && a !== "upload").map((a) => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
        </div>
        <div className="appbar-item appbar-item--mobile">
          <label htmlFor="mapping-select-m">Mapping</label>
          <select
            id="mapping-select-m"
            className="select"
            value={strategy}
            onChange={(e) => setStrategy(e.target.value as MappingStrategy)}
            aria-label="Select mapping strategy"
          >
            <option value="direct">direct</option>
            <option value="mel">mel</option>
            <option value="energy">energy</option>
          </select>
        </div>
        <div className="appbar-item appbar-item--mobile">
          <label htmlFor="contour-select-m">Contours</label>
          <select
            id="contour-select-m"
            className="select"
            value={contourLevels}
            onChange={(e) => setContourLevels(Number(e.target.value))}
            aria-label="Contour levels"
          >
            {[0, 3, 5, 7, 9, 12, 16].map((n) => (
              <option key={n} value={n}>{n === 0 ? "None" : `${n} levels`}</option>
            ))}
          </select>
        </div>
      </div>
      <div className="appbar-right appbar-right--mobile">
        <div className="appbar-right__cell">
          <button
            type="button"
            className={"appbar-btn appbar-btn--full" + (audioStimulus === "mic" ? " appbar-btn--active" : "")}
            onClick={() => setAudioStimulus("mic")}
            aria-pressed={audioStimulus === "mic"}
            aria-label="Live microphone"
          >
            Live Mic
          </button>
          {audioStimulus === "mic" && (
            <label className="checkbox-wrap checkbox-wrap--inline">
              <input type="checkbox" checked={micOn} onChange={(e) => setMicOn(e.target.checked)} aria-label="Use microphone" />
              <span className="checkbox-wrap__label">On</span>
            </label>
          )}
        </div>
        <div className="appbar-right__cell">
          <button
            type="button"
            className={"appbar-btn appbar-btn--full" + (audioStimulus === "upload" ? " appbar-btn--active" : "")}
            onClick={() => setAudioStimulus("upload")}
            aria-pressed={audioStimulus === "upload"}
            aria-label="Upload audio file"
          >
            Upload
          </button>
          {audioStimulus === "upload" && onUploadFileChange && (
            <label className="appbar-upload appbar-upload--trigger appbar-upload--mobile appbar-upload--full">
              <input
                type="file"
                accept="audio/*"
                className="appbar-upload__input"
                onChange={(e) => onUploadFileChange(e.target.files?.[0] ?? null)}
                aria-label="Choose audio file"
              />
              <span className="appbar-upload__text">
                {uploadedFileName ? (uploadedFileName.length > 18 ? uploadedFileName.slice(0, 15) + "…" : uploadedFileName) : "Choose file"}
              </span>
            </label>
          )}
        </div>
      </div>
    </header>
  );
}
