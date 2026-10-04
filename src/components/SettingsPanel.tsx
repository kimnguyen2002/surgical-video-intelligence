/**
 * Settings: the assistant's audience, and the optional language model.
 *
 * This panel is where the project's cost model is explained to the person
 * using it, which is why it leads with what works *without* a key rather than
 * with the key field. A visitor should be able to close it having learned that
 * they do not need one.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { Check, ExternalLink, Eye, EyeOff, Key, Loader2, Trash2, X } from 'lucide-react';

import { ASSISTANT_ROLES } from '../constants';
import { apiKey, checkKey, listModels, preferredModel, type GeminiModel } from '../services/byokGemini';
import type { AssistantRole } from '../types';

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  role: AssistantRole;
  onRoleChange: (role: AssistantRole) => void;
  useGenerative: boolean;
  onUseGenerativeChange: (value: boolean) => void;
  modelId: string;
  onModelChange: (modelId: string) => void;
  /** Re-probes the key and refreshes the banner in the parent. */
  onKeyChanged: () => void;
}

type KeyState =
  | { kind: 'absent' }
  | { kind: 'checking' }
  | { kind: 'valid'; model: string }
  | { kind: 'invalid'; message: string; hint?: string };

export function SettingsPanel({
  open,
  onClose,
  role,
  onRoleChange,
  useGenerative,
  onUseGenerativeChange,
  modelId,
  onModelChange,
  onKeyChanged,
}: SettingsPanelProps) {
  const [draftKey, setDraftKey] = useState('');
  const [revealed, setRevealed] = useState(false);
  const [keyState, setKeyState] = useState<KeyState>(
    apiKey.configured ? { kind: 'checking' } : { kind: 'absent' }
  );
  const [models, setModels] = useState<GeminiModel[]>([]);

  const probe = useCallback(async () => {
    if (!apiKey.configured) {
      setKeyState({ kind: 'absent' });
      setModels([]);
      return;
    }
    setKeyState({ kind: 'checking' });
    const result = await checkKey();
    if (!result.ok) {
      setKeyState({ kind: 'invalid', message: result.failure.message, hint: result.failure.hint });
      setModels([]);
      return;
    }
    setKeyState({ kind: 'valid', model: result.model });
    try {
      setModels(await listModels());
    } catch {
      // The key works for generation but the catalogue call failed; the model
      // it resolved to is still usable, so this is not worth surfacing.
      setModels([]);
    }
  }, []);

  useEffect(() => {
    if (open) void probe();
  }, [open, probe]);

  if (!open) return null;

  const saveKey = async () => {
    const trimmed = draftKey.trim();
    if (!trimmed) return;
    apiKey.set(trimmed);
    // A newly pasted key invalidates whatever model the previous one resolved
    // to; leaving it set would send the first request to a model this key may
    // not have.
    preferredModel.set('');
    setDraftKey('');
    setRevealed(false);
    await probe();
    onKeyChanged();
  };

  const clearKey = async () => {
    apiKey.clear();
    preferredModel.set('');
    onUseGenerativeChange(false);
    setKeyState({ kind: 'absent' });
    setModels([]);
    onKeyChanged();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/70 p-4 overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="mt-10 w-full max-w-xl rounded-lg border border-line bg-card shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex items-center justify-between border-b border-line px-5 py-3">
          <h2 className="text-sm font-semibold text-fg">Settings</h2>
          <button
            onClick={onClose}
            className="rounded p-1 text-fg2 transition-colors hover:bg-hover hover:text-fg"
            aria-label="Close settings"
          >
            <X className="h-4 w-4" />
          </button>
        </header>

        <div className="space-y-6 px-5 py-5">
          {/* ---- Audience ---- */}
          <section>
            <h3 className="text-xs font-semibold text-fg2">
              Answer for
            </h3>
            <p className="mt-1 text-xs text-fg2">
              Changes how much is explained and in what register — a resident and a
              computer-vision researcher want different answers to the same question.
            </p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {ASSISTANT_ROLES.map((r) => (
                <button
                  key={r}
                  onClick={() => onRoleChange(r)}
                  className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
                    role === r
                      ? 'border-accent/40 bg-accent-soft font-semibold text-accent'
                      : 'border-line text-fg2 hover:bg-hover'
                  }`}
                >
                  {r}
                </button>
              ))}
            </div>
          </section>

          {/* ---- The cost model, stated plainly ---- */}
          <section className="rounded-lg border border-ok-line bg-ok-soft p-4">
            <h3 className="flex items-center gap-2 text-xs font-semibold text-ok">
              <Check className="h-3.5 w-3.5" />
              No key needed
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-fg2">
              Instrument detection, presence and task recognition, the activation heatmap and the assistant all
              run <strong>in this browser</strong>. Nothing is uploaded, no server is involved, and
              there is no API to run out of. The assistant answers questions about the video from
              the SurgVU release&rsquo;s own annotations — which, for &ldquo;what instrument is
              installed?&rdquo;, is a better answer than any language model would give.
            </p>
          </section>

          {/* ---- Optional BYO key ---- */}
          <section>
            <h3 className="flex items-center gap-2 text-xs font-semibold text-fg2">
              <Key className="h-3.5 w-3.5" />
              Language model (optional)
            </h3>
            <p className="mt-1 text-xs leading-relaxed text-fg2">
              For generated explanations — <em>why</em> a step is sequenced this way, what the
              trade-offs are — add your own Gemini API key. It is stored in this browser only and
              sent directly to Google; it never reaches any server belonging to this project,
              because the project does not have one.
            </p>

            {keyState.kind === 'absent' && (
              <div className="mt-3 space-y-2">
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <input
                      type={revealed ? 'text' : 'password'}
                      value={draftKey}
                      onChange={(event) => setDraftKey(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') void saveKey();
                      }}
                      placeholder="AIza…"
                      autoComplete="off"
                      spellCheck={false}
                      className="w-full rounded-md border border-line bg-page px-3 py-2 pr-9 font-mono text-xs text-fg outline-none focus:border-accent"
                    />
                    <button
                      onClick={() => setRevealed((v) => !v)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-fg2 hover:text-fg2"
                      aria-label={revealed ? 'Hide key' : 'Show key'}
                    >
                      {revealed ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                    </button>
                  </div>
                  <button
                    onClick={() => void saveKey()}
                    disabled={!draftKey.trim()}
                    className="rounded-md bg-accent px-3 py-2 text-xs font-semibold text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Save
                  </button>
                </div>
                <a
                  href="https://aistudio.google.com/apikey"
                  target="_blank"
                  rel="noreferrer noopener"
                  className="inline-flex items-center gap-1 text-xs text-accent hover:underline"
                >
                  Get a free Gemini API key
                  <ExternalLink className="h-3 w-3" />
                </a>
              </div>
            )}

            {keyState.kind === 'checking' && (
              <p className="mt-3 flex items-center gap-2 text-xs text-fg2">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Checking the key…
              </p>
            )}

            {keyState.kind === 'invalid' && (
              <div className="mt-3 space-y-2 rounded-md border border-warn-line bg-warn-soft p-3">
                <p className="text-xs text-warn">{keyState.message}</p>
                {keyState.hint && <p className="text-xs text-fg2">{keyState.hint}</p>}
                <button
                  onClick={() => void clearKey()}
                  className="inline-flex items-center gap-1.5 text-xs text-fg2 hover:text-risk"
                >
                  <Trash2 className="h-3 w-3" />
                  Remove this key
                </button>
              </div>
            )}

            {keyState.kind === 'valid' && (
              <div className="mt-3 space-y-3">
                <div className="flex items-center justify-between rounded-md border border-ok-line bg-ok-soft px-3 py-2">
                  <span className="flex items-center gap-2 text-xs text-ok">
                    <Check className="h-3.5 w-3.5" />
                    Key verified
                  </span>
                  <button
                    onClick={() => void clearKey()}
                    className="inline-flex items-center gap-1.5 text-xs text-fg2 hover:text-risk"
                  >
                    <Trash2 className="h-3 w-3" />
                    Remove
                  </button>
                </div>

                <label className="flex cursor-pointer items-start gap-2.5">
                  <input
                    type="checkbox"
                    checked={useGenerative}
                    onChange={(event) => onUseGenerativeChange(event.target.checked)}
                    className="mt-0.5 h-3.5 w-3.5 accent-cyan-500"
                  />
                  <span className="text-xs leading-relaxed text-fg2">
                    Use generated answers for open-ended questions.
                    <span className="mt-0.5 block text-xs text-fg2">
                      Factual lookups — which instrument, how many, when — stay on the grounded
                      path either way. Paraphrasing a dataset value through a model cannot make it
                      more correct, can make it less, and costs a request.
                    </span>
                  </span>
                </label>

                {models.length > 0 && (
                  <label className="block">
                    <span className="text-xs text-fg2">Model</span>
                    <select
                      value={modelId || keyState.model}
                      onChange={(event) => {
                        preferredModel.set(event.target.value);
                        onModelChange(event.target.value);
                      }}
                      className="mt-1 w-full rounded-md border border-line bg-page px-2 py-1.5 text-xs text-fg outline-none focus:border-accent"
                    >
                      {models.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.displayName}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>
            )}

            <p className="mt-3 text-xs leading-relaxed text-fg2">
              A key in browser storage is readable by any script on this origin. That is acceptable
              for a key scoped to the Generative Language API and nothing else; it is not
              acceptable for one with broader Google Cloud permissions.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
