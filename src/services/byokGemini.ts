/**
 * Optional generative answers, using a key the *visitor* supplies.
 *
 * Why it works this way
 * ---------------------
 * The previous build ran a server that held the author's `GEMINI_API_KEY` and
 * proxied every call. That is the correct design for a product and the wrong
 * one for a public demo: every visitor spends the author's quota, a loop left
 * running overnight is an open-ended bill, and the key is one misconfigured
 * environment variable away from being public.
 *
 * So there is no server and no key in this repository. The app is complete
 * without one — the grounded assistant in `src/assistant/` answers from the
 * dataset's own annotations at zero cost and works offline. This module is the
 * *upgrade path*: a visitor who wants generated prose pastes their own key,
 * and it is their quota that is spent.
 *
 * Where the key goes
 * ------------------
 * Into `localStorage`, on their machine, and into the `x-goog-api-key` header
 * of requests that go **directly** from their browser to Google. It never
 * reaches a server of ours, because there isn't one. A static site cannot
 * exfiltrate what it never receives, and that is a property of the
 * architecture rather than a promise in a privacy policy.
 *
 * The honest caveat, stated in the settings panel too: a key in `localStorage`
 * is readable by any script running on this origin. That is acceptable for a
 * demo key scoped to the Generative Language API and nothing else, and it is
 * not acceptable for a key with broader Cloud permissions. The panel says so
 * and links to the page where a restricted key is made.
 */

const STORAGE_KEY = 'svi.gemini.apiKey';
const STORAGE_MODEL = 'svi.gemini.model';

const API_ROOT = 'https://generativelanguage.googleapis.com/v1beta';

export interface GeminiFailure {
  code: number;
  status: string;
  message: string;
  hint?: string;
}

export class GeminiError extends Error {
  readonly failure: GeminiFailure;
  constructor(failure: GeminiFailure) {
    super(failure.message);
    this.name = 'GeminiError';
    this.failure = failure;
  }
}

// ---------------------------------------------------------------------------
// Key storage
// ---------------------------------------------------------------------------

/**
 * Storage access is wrapped because it throws rather than returning null in a
 * private window and in some embedded contexts. An assistant that crashes the
 * page because the browser declined to remember a key is worse than one that
 * quietly runs without it.
 */
function readStored(key: string): string {
  try {
    return localStorage.getItem(key) || '';
  } catch {
    return '';
  }
}

function writeStored(key: string, value: string): void {
  try {
    if (value) localStorage.setItem(key, value);
    else localStorage.removeItem(key);
  } catch {
    // Non-fatal: the key lives for this page load only.
  }
}

export const apiKey = {
  get: () => readStored(STORAGE_KEY).trim(),
  set: (value: string) => writeStored(STORAGE_KEY, value.trim()),
  clear: () => writeStored(STORAGE_KEY, ''),
  get configured() {
    return readStored(STORAGE_KEY).trim().length > 0;
  },
};

export const preferredModel = {
  get: () => readStored(STORAGE_MODEL).trim(),
  set: (value: string) => writeStored(STORAGE_MODEL, value.trim()),
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/**
 * Turn an API error into something the reader can act on.
 *
 * A 403 here almost never means "try again", which is what a generic error
 * banner implies. It means the key is restricted, the Generative Language API
 * is off for that project, or the key belongs to a project without billing —
 * all of which need a human, so the hint says which.
 */
function describe(status: number, payload: unknown): GeminiFailure {
  const inner = (payload as { error?: Record<string, unknown> })?.error ?? {};
  const code = Number(inner.code) || status || 500;
  const apiStatus = String(inner.status || 'ERROR');
  const message = String(inner.message || `Request failed (${status}).`);

  let hint: string | undefined;
  if (code === 403 || apiStatus === 'PERMISSION_DENIED') {
    hint =
      'The key reached Google but was rejected. Check, in order: (1) it is a Gemini API key ' +
      'from aistudio.google.com/apikey, not a Cloud console OAuth client; (2) the Generative ' +
      'Language API is enabled on that key’s project; (3) any HTTP-referrer restriction on ' +
      'the key includes this site’s origin.';
  } else if (code === 400 && /api.?key not valid/i.test(message)) {
    // The API answers a malformed key with 400 INVALID_ARGUMENT rather than
    // 401, so matching on the code alone leaves the commonest mistake — a
    // truncated or wrongly-pasted key — with no advice attached.
    hint =
      'The key was rejected as malformed. Re-copy it from aistudio.google.com/apikey and paste ' +
      'it with no surrounding quotes or whitespace.';
  } else if (code === 429 || apiStatus === 'RESOURCE_EXHAUSTED') {
    hint =
      'Your key’s quota is exhausted. The grounded assistant keeps working without it — it ' +
      'does not call any API.';
  } else if (code === 404 || apiStatus === 'NOT_FOUND') {
    hint = 'That model is not available to this key. Pick another in Settings.';
  }

  return { code, status: apiStatus, message, hint };
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const key = apiKey.get();
  if (!key) {
    throw new GeminiError({
      code: 401,
      status: 'NO_KEY',
      message: 'No Gemini API key is configured in this browser.',
      hint: 'Add one under Settings → Language model, or keep using the grounded assistant.',
    });
  }

  let response: Response;
  try {
    response = await fetch(`${API_ROOT}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        // The key goes in a header rather than the query string so it does not
        // land in any intermediary's request log or in the browser's history.
        'x-goog-api-key': key,
        ...(init.headers || {}),
      },
    });
  } catch (err) {
    throw new GeminiError({
      code: 0,
      status: 'NETWORK',
      message: err instanceof Error ? err.message : 'The request could not be sent.',
      hint: 'Check the network connection. The grounded assistant works offline.',
    });
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new GeminiError(describe(response.status, payload));
  return payload as T;
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export interface GeminiModel {
  id: string;
  displayName: string;
}

/**
 * The text models this key can actually call.
 *
 * Hardcoding a list of model ids would break the moment one is retired, and
 * the failure would read as "the assistant is broken" rather than "that model
 * is gone". Asking the key what it has means the app follows Google's
 * catalogue without a release.
 */
export async function listModels(): Promise<GeminiModel[]> {
  const payload = await call<{
    models?: { name?: string; displayName?: string; supportedGenerationMethods?: string[] }[];
  }>('/models?pageSize=200');

  return (payload.models || [])
    .filter((m) => m.supportedGenerationMethods?.includes('generateContent'))
    .map((m) => ({
      id: (m.name || '').replace(/^models\//, ''),
      displayName: m.displayName || (m.name || '').replace(/^models\//, ''),
    }))
    .filter((m) => m.id && !/embedding|aqa|imagen|veo/i.test(m.id))
    // Flash models first: they are the cheapest and fastest, which is what a
    // visitor spending their own quota on a demo wants by default.
    .sort((a, b) => {
      const rank = (id: string) => (/flash/i.test(id) ? 0 : /pro/i.test(id) ? 1 : 2);
      return rank(a.id) - rank(b.id) || a.id.localeCompare(b.id);
    });
}

/** The model to use: the visitor's choice, else the first flash model found. */
export async function resolveModel(): Promise<string> {
  const chosen = preferredModel.get();
  if (chosen) return chosen;

  const models = await listModels();
  if (models.length === 0) {
    throw new GeminiError({
      code: 404,
      status: 'NO_MODELS',
      message: 'This key has no text-generation models available.',
      hint: 'Check that the Generative Language API is enabled on the key’s project.',
    });
  }
  preferredModel.set(models[0].id);
  return models[0].id;
}

/** Whether the stored key actually works, as opposed to merely existing. */
export async function checkKey(): Promise<{ ok: true; model: string } | { ok: false; failure: GeminiFailure }> {
  try {
    const model = await resolveModel();
    return { ok: true, model };
  } catch (err) {
    if (err instanceof GeminiError) return { ok: false, failure: err.failure };
    throw err;
  }
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

export interface ChatTurn {
  role: 'user' | 'model';
  text: string;
}

/**
 * One generated reply.
 *
 * `systemInstruction` carries the grounding: the recorded annotations and the
 * detector output for the current frame, plus the standing rule that recorded
 * and predicted claims are never blurred together. The model is being asked to
 * *explain* facts the app already established, not to look at the video and
 * decide what is in it — nothing here sends an image, and no per-frame call is
 * ever made. That is what keeps a visitor's quota spend proportional to the
 * questions they ask rather than to how long they leave the video playing.
 */
export async function generate(
  turns: ChatTurn[],
  systemInstruction: string,
  signal?: AbortSignal
): Promise<{ text: string; model: string }> {
  const model = await resolveModel();

  const payload = await call<{
    candidates?: { content?: { parts?: { text?: string }[] }; finishReason?: string }[];
    promptFeedback?: { blockReason?: string };
  }>(`/models/${encodeURIComponent(model)}:generateContent`, {
    method: 'POST',
    signal,
    body: JSON.stringify({
      contents: turns.slice(-12).map((t) => ({ role: t.role, parts: [{ text: t.text }] })),
      systemInstruction: { parts: [{ text: systemInstruction }] },
      generationConfig: { temperature: 0.3, maxOutputTokens: 1024 },
    }),
  });

  if (payload.promptFeedback?.blockReason) {
    throw new GeminiError({
      code: 400,
      status: 'BLOCKED',
      message: `The request was blocked (${payload.promptFeedback.blockReason}).`,
      hint: 'Rephrase the question. The grounded assistant can still answer from the annotations.',
    });
  }

  const text = (payload.candidates?.[0]?.content?.parts || [])
    .map((p) => p.text || '')
    .join('')
    .trim();

  if (!text) {
    throw new GeminiError({
      code: 502,
      status: 'EMPTY',
      message: 'The model returned no text.',
      hint: 'Retry, or pick a different model in Settings.',
    });
  }

  return { text, model };
}
