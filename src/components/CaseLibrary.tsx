import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  FolderOpen,
  FileVideo,
  CheckCircle2,
  ChevronRight,
  ChevronDown,
  Database,
  HardDrive,
  Wrench,
  ListChecks,
  Scissors,
  Sparkles,
} from 'lucide-react';
import { SURGVU_CASES } from '../data/surgvuCases';
import { SURGVU_CLIPS, SURGVU_CLIPS_TOTAL_BYTES, clipFor } from '../data/surgvuClips';
import { formatBytes, formatClock, matchFileToPart } from '../services/groundTruth';
import { toolDisplay } from '../data/surgvuVocab';

/** Key a case/part pair the way the attached-file map does. */
export const partId = (caseId: string, part: number) => `${caseId}/${part}`;

interface CaseLibraryProps {
  attachedFiles: Map<string, File>;
  activePartId: string | null;
  onAttachFiles: (files: File[]) => void;
  onSelectPart: (caseId: string, part: number) => void;
}

/**
 * Browse the SurgVU cases: play the bundled excerpt, or attach the real files
 * or bind local disk video files for zero-latency playback.
 */
export const CaseLibrary: React.FC<CaseLibraryProps> = ({
  attachedFiles,
  activePartId,
  onAttachFiles,
  onSelectPart,
}) => {
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set(['000']));
  const [rejected, setRejected] = useState<string[]>([]);

  // `webkitdirectory` is not in React's JSX attribute types, so it is set on
  // the DOM node directly. Browsers without it simply get the file picker.
  useEffect(() => {
    const input = folderInputRef.current;
    if (input) {
      input.setAttribute('webkitdirectory', '');
      input.setAttribute('directory', '');
    }
  }, []);

  const totals = useMemo(() => {
    const parts = SURGVU_CASES.flatMap((c) => c.parts);
    return {
      cases: SURGVU_CASES.length,
      parts: parts.length,
      seconds: parts.reduce((sum, p) => sum + p.durationSeconds, 0),
      bytes: parts.reduce((sum, p) => sum + p.sizeBytes, 0),
      intervals: SURGVU_CASES.reduce((sum, c) => sum + c.toolIntervalCount + c.taskIntervalCount, 0),
    };
  }, []);

  const handleFiles = (fileList: FileList | null) => {
    if (!fileList) return;
    const picked = Array.from(fileList);
    const accepted = picked.filter((f) => matchFileToPart(f.name));
    const unknown = picked
      .filter((f) => /\.(mp4|webm|mov)$/i.test(f.name) && !matchFileToPart(f.name))
      .map((f) => f.name);

    setRejected(unknown);
    if (accepted.length) onAttachFiles(accepted);

    // Auto-open the cases that just gained a file.
    setExpanded((prev) => {
      const next = new Set(prev);
      for (const file of accepted) {
        const match = matchFileToPart(file.name);
        if (match) next.add(match.caseId);
      }
      return next;
    });
  };

  const toggleCase = (caseId: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(caseId)) next.delete(caseId);
      else next.add(caseId);
      return next;
    });
  };

  const attachedCount = attachedFiles.size;
  const excerptCount = SURGVU_CLIPS.length;

  return (
    <div className="bg-card p-3.5 rounded-lg border border-line flex flex-col space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-[13px] font-semibold text-fg">
          <Database className="w-3.5 h-3.5 text-ok" />
          <span>SurgVU case library</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-fg2 font-mono">
            {attachedCount > 0
              ? `${attachedCount} attached · ${excerptCount} excerpts`
              : `${excerptCount} excerpts · ${totals.parts} parts`}
          </span>
        </div>
      </div>

      {/* What you are actually about to play, and what you are not. */}
      <div className="bg-card border border-line rounded-md p-2.5 flex items-start justify-between gap-2">
        <div className="flex items-start gap-2 min-w-0">
          <div className="p-1 rounded bg-ok-soft text-ok flex-shrink-0 mt-0.5">
            <Scissors className="w-3.5 h-3.5" />
          </div>
          <div className="min-w-0">
            <div className="text-xs font-medium text-fg">
              {excerptCount} excerpts bundled ({formatBytes(SURGVU_CLIPS_TOTAL_BYTES)})
            </div>
            <div className="text-[11px] text-fg2 mt-0.5 leading-relaxed">
              Two minutes per part, cut from its most densely-labelled window — not from the start,
              where nothing is installed and nothing is labelled. Timestamps shown are the real
              ones from the full recording.
            </div>
          </div>
        </div>
        <span className="text-[11px] text-ok bg-ok-soft border border-ok-line px-2 py-1 rounded whitespace-nowrap flex-shrink-0">
          No upload
        </span>
      </div>

      {/* Dataset summary */}
      <div className="grid grid-cols-4 gap-2 text-center">
        {[
          { icon: FileVideo, value: `${totals.cases}`, label: 'cases' },
          { icon: HardDrive, value: formatBytes(totals.bytes), label: `${totals.parts} parts` },
          { icon: ListChecks, value: formatClock(totals.seconds).split(':')[0] + ' h', label: 'footage' },
          { icon: Wrench, value: `${totals.intervals}`, label: 'labelled intervals' },
        ].map((stat) => (
          <div key={stat.label} className="bg-card border border-line rounded-md py-1.5 px-1">
            <stat.icon className="w-3 h-3 text-fg2 mx-auto mb-0.5" />
            <div className="text-[13px] font-semibold text-fg leading-none">{stat.value}</div>
            <div className="text-[11px] text-fg2 mt-0.5">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Attach controls (Optional local override) */}
      <input
        ref={folderInputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = '';
        }}
      />
      <input
        ref={fileInputRef}
        type="file"
        accept="video/mp4,video/webm,video/quicktime"
        multiple
        className="hidden"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = '';
        }}
      />

      <div className="flex items-center justify-between gap-2 pt-0.5">
        <span className="text-xs font-medium text-fg2">
          Select any case below to stream, or override with local disk files:
        </span>
        <div className="flex gap-1.5 flex-shrink-0">
          <button
            id="btn-attach-folder"
            onClick={() => folderInputRef.current?.click()}
            className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-fg2 bg-inset hover:bg-hover rounded transition-colors border border-line"
            title="Pick a local folder if you prefer zero-network local playback"
          >
            <FolderOpen className="w-3 h-3 text-fg2" />
            <span>Attach folder</span>
          </button>

          <button
            id="btn-attach-files"
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-fg2 bg-inset hover:bg-hover rounded transition-colors border border-line"
            title="Attach a single local mp4 file"
          >
            <FileVideo className="w-3 h-3 text-fg2" />
            <span>Single file</span>
          </button>
        </div>
      </div>

      {rejected.length > 0 && (
        <div className="text-[11px] text-warn bg-warn-soft border border-warn-line rounded p-2 leading-snug">
          Not in the SurgVU manifest, so not attached (no labels exist for them):{' '}
          <span className="font-mono">{rejected.slice(0, 3).join(', ')}</span>
          {rejected.length > 3 && ` and ${rejected.length - 3} more`}.
        </div>
      )}

      {/* Case list */}
      <div className="flex flex-col gap-1 max-h-80 overflow-y-auto pr-1">
        {SURGVU_CASES.map((surgCase) => {
          const isOpen = expanded.has(surgCase.caseId);
          const attachedInCase = surgCase.parts.filter((p) =>
            attachedFiles.has(partId(surgCase.caseId, p.part))
          ).length;

          return (
            <div key={surgCase.caseId} className="rounded-md border border-line bg-card">
              <button
                onClick={() => toggleCase(surgCase.caseId)}
                className="w-full flex items-center justify-between gap-2 px-2.5 py-2 hover:bg-hover transition-colors rounded-md"
              >
                <div className="flex items-center gap-2 min-w-0">
                  {isOpen ? (
                    <ChevronDown className="w-3.5 h-3.5 text-fg2 flex-shrink-0" />
                  ) : (
                    <ChevronRight className="w-3.5 h-3.5 text-fg2 flex-shrink-0" />
                  )}
                  <span className="text-xs font-semibold text-fg">
                    case_{surgCase.caseId}
                  </span>
                  <span className="text-[11px] text-fg2 font-mono">
                    {formatClock(surgCase.totalDurationSeconds)} · {surgCase.parts.length}{' '}
                    {surgCase.parts.length === 1 ? 'part' : 'parts'}
                  </span>
                </div>

                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <span className="text-[11px] text-fg2">
                    {surgCase.toolIntervalCount} tool · {surgCase.taskIntervalCount} task
                  </span>
                  {attachedInCase > 0 ? (
                    <span className="text-[11px] text-ok bg-ok-soft border border-ok-line px-1.5 py-0.2 rounded">
                      Local
                    </span>
                  ) : surgCase.parts.some((p) => clipFor(surgCase.caseId, p.part)) ? (
                    <span className="text-[11px] text-accent bg-accent-soft border border-accent/40 px-1.5 py-0.2 rounded">
                      Excerpt
                    </span>
                  ) : null}
                </div>
              </button>

              {isOpen && (
                <div className="px-2.5 pb-2 flex flex-col gap-1.5">
                  {/* Tool classes recorded in this case */}
                  <div className="flex flex-wrap gap-1 pb-1">
                    {surgCase.toolClasses.map((cls) => (
                      <span
                        key={cls}
                        className="text-[11px] px-1.5 py-0.5 rounded bg-inset text-fg2 border border-line"
                      >
                        {toolDisplay(cls)}
                      </span>
                    ))}
                  </div>

                  {surgCase.parts.map((part) => {
                    const key = partId(surgCase.caseId, part.part);
                    const file = attachedFiles.get(key);
                    const isActive = activePartId === key;
                    // A part is playable when the visitor attached the real
                    // file, or when an excerpt was cut from this exact part.
                    const excerpt = clipFor(surgCase.caseId, part.part);
                    const hasExcerpt = Boolean(excerpt);
                    const isAvailable = Boolean(file || hasExcerpt);

                    return (
                      <button
                        key={key}
                        id={`library-part-${key.replace('/', '-')}`}
                        disabled={!isAvailable}
                        onClick={() => onSelectPart(surgCase.caseId, part.part)}
                        className={`w-full text-left px-2 py-1.5 rounded border text-xs transition-colors flex items-center justify-between gap-2 ${
                          isActive
                            ? 'border-accent bg-accent-soft text-accent shadow-sm'
                            : file
                              ? 'border-line bg-inset text-fg hover:border-ok-line hover:bg-ok-soft cursor-pointer'
                              : hasExcerpt
                                ? 'border-line bg-inset text-fg hover:border-accent/40 hover:bg-accent-soft cursor-pointer'
                                : 'border-dashed border-line bg-transparent text-fg2 cursor-not-allowed'
                        }`}
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          {file ? (
                            <HardDrive className="w-3 h-3 text-ok flex-shrink-0" />
                          ) : hasExcerpt ? (
                            <Scissors className="w-3 h-3 text-accent flex-shrink-0" />
                          ) : (
                            <FileVideo className="w-3 h-3 text-fg2 flex-shrink-0" />
                          )}
                          <span className="font-mono truncate">{part.filename}</span>
                        </div>
                        <span className="flex items-center gap-1.5 flex-shrink-0">
                          <span className="text-[11px] opacity-80">
                            {formatClock(part.durationSeconds)} · {formatBytes(part.sizeBytes)}
                          </span>
                          {file ? (
                            <span className="text-[11px] px-1.5 py-0.5 rounded bg-ok-soft text-ok border border-ok-line">
                              Local
                            </span>
                          ) : hasExcerpt ? (
                            <span className="text-[11px] px-1.5 py-0.5 rounded bg-accent-soft text-accent border border-accent/40">
                              {excerpt!.durationSeconds}s excerpt
                            </span>
                          ) : (
                            <span className="text-[11px] italic text-fg2">not attached</span>
                          )}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
