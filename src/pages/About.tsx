import React, { useEffect, useRef, useState } from 'react';
import {
  Stethoscope,
  ArrowRight,
  ArrowUpRight,
  FileText,
  Database,
  Cpu,
  Workflow,
  ShieldCheck,
  ShieldAlert,
  Eye,
  Github,
  GraduationCap,
  Globe,
  Layers,
  Timer,
  Gauge,
  Boxes,
  AlertTriangle,
  Quote,
} from 'lucide-react';
import { SURGVU_CASES } from '../data/surgvuCases';
import { ThemeToggle } from '../components/ThemeToggle';
import {
  TOOL_CLASSES,
  TASK_CLASSES,
  toolDisplay,
  taskDisplay,
} from '../data/surgvuVocab';

/**
 * The About page.
 *
 * Everything the console shows in fragments — which release the video comes
 * from, which checkpoints are running, what their numbers actually mean — is
 * stated once, here, in full. The numbers about *this build* are computed from
 * the same generated manifest the player reads, so the page cannot drift from
 * the data it describes: regenerate the labels and these figures follow.
 */

/* ------------------------------------------------------------------ */
/* Figures derived from the bundled manifest, not typed in by hand.    */
/* ------------------------------------------------------------------ */

const BUNDLED = (() => {
  const tools = new Set<string>();
  const tasks = new Set<string>();
  let parts = 0;
  let seconds = 0;
  let bytes = 0;
  let toolIntervals = 0;
  let taskIntervals = 0;

  for (const c of SURGVU_CASES) {
    parts += c.parts.length;
    seconds += c.totalDurationSeconds;
    toolIntervals += c.toolIntervalCount;
    taskIntervals += c.taskIntervalCount;
    c.toolClasses.forEach((t) => tools.add(t));
    c.taskClasses.forEach((t) => tasks.add(t));
    c.parts.forEach((p) => (bytes += p.sizeBytes));
  }

  return {
    cases: SURGVU_CASES.length,
    parts,
    hours: seconds / 3600,
    gigabytes: bytes / 1e9,
    toolIntervals,
    taskIntervals,
    toolClasses: tools.size,
    taskClasses: tasks.size,
  };
})();

/* ------------------------------------------------------------------ */
/* Small building blocks                                              */
/* ------------------------------------------------------------------ */

/** Fades a block in the first time it scrolls into view; inert if the user asked for less motion. */
const Reveal: React.FC<{
  children: React.ReactNode;
  className?: string;
  delay?: number;
  as?: 'div' | 'section' | 'li';
}> = ({ children, className = '', delay = 0, as = 'div' }) => {
  const ref = useRef<HTMLElement>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setShown(true);
          io.disconnect();
        }
      },
      { rootMargin: '0px 0px -10% 0px', threshold: 0.05 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const Tag = as as any;
  return (
    <Tag
      ref={ref as any}
      className={`reveal ${shown ? 'is-visible' : ''} ${className}`}
      style={delay ? { transitionDelay: `${delay}ms` } : undefined}
    >
      {children}
    </Tag>
  );
};

const Section: React.FC<{ id: string; children: React.ReactNode; className?: string }> = ({
  id,
  children,
  className = '',
}) => (
  <section id={id} className={`scroll-mt-24 py-16 sm:py-24 ${className}`}>
    <div className="max-w-[1120px] mx-auto px-6">{children}</div>
  </section>
);

const SectionHead: React.FC<{
  index: string;
  kicker: string;
  title: string;
  blurb?: React.ReactNode;
  icon?: React.ReactNode;
}> = ({ index, kicker, title, blurb, icon }) => (
  <Reveal className="mb-10 sm:mb-14">
    <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-accent" data-index={index}>
      {icon}
      <span>{kicker}</span>
    </div>
    <h2 className="mt-4 text-[28px] sm:text-[42px] font-extrabold tracking-tight text-fg leading-[1.1]">
      {title}
    </h2>
    {blurb && (
      <p className="mt-4 max-w-3xl text-[15px] leading-relaxed text-fg2">{blurb}</p>
    )}
  </Reveal>
);

const Card: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = '',
}) => (
  <div
    className={`rounded-lg border border-line bg-card p-5 transition-colors hover:border-line-strong ${className}`}
  >
    {children}
  </div>
);

const Chip: React.FC<{ children: React.ReactNode; tone?: 'slate' | 'cyan' | 'emerald' }> = ({
  children,
  tone = 'slate',
}) => {
  const tones = {
    slate: 'border-line bg-transparent text-fg3',
    cyan: 'border-accent/40 bg-accent-soft text-accent',
    emerald: 'border-ok-line bg-ok-soft text-ok',
  } as const;
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
};

const Stat: React.FC<{ value: string; unit?: string; label: string; note?: string }> = ({
  value,
  unit,
  label,
  note,
}) => (
  <div className="rounded-lg border border-line bg-card px-5 py-4">
    <div className="flex items-baseline gap-1">
      <span className="text-2xl sm:text-[30px] font-extrabold text-fg tracking-tight">
        {value}
      </span>
      {unit && <span className="text-xs font-semibold text-accent">{unit}</span>}
    </div>
    <div className="mt-1 text-[12px] font-medium text-fg2">{label}</div>
    {note && <div className="mt-0.5 text-[11px] text-fg3 leading-snug">{note}</div>}
  </div>
);

/** A label/value row in the mono spec lists. */
const Spec: React.FC<{ k: string; v: React.ReactNode }> = ({ k, v }) => (
  <div className="flex items-start justify-between gap-6 border-b border-line py-2.5 last:border-0">
    <span className="text-[12px] text-fg3">{k}</span>
    <span className="text-right text-[12px] font-mono text-fg">{v}</span>
  </div>
);

const NAV = [
  { id: 'why', label: 'Why' },
  { id: 'dataset', label: 'Dataset' },
  { id: 'models', label: 'Models' },
  { id: 'pipeline', label: 'Pipeline' },
  { id: 'provenance', label: 'Provenance' },
  { id: 'author', label: 'Author' },
];

/* ------------------------------------------------------------------ */

export default function About() {
  const [active, setActive] = useState<string>('why');

  // Scroll spy: the nav says where you are, without touching the hash — the
  // hash is the router's, and an anchor link would navigate away from /about.
  useEffect(() => {
    const els = NAV.map((n) => document.getElementById(n.id)).filter(Boolean) as HTMLElement[];
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: '-20% 0px -65% 0px', threshold: 0 }
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);

  const goTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <div className="min-h-screen bg-page text-fg font-sans selection:bg-accent selection:text-on-accent">
      {/* ---------------------------------------------------------------- */}
      {/* NAV                                                              */}
      {/* ---------------------------------------------------------------- */}
      <header className="sticky top-0 z-50 border-b border-line bg-page/85 backdrop-blur-md">
        <div className="max-w-[1120px] mx-auto px-6 h-14 flex items-center justify-between gap-4">
          <a href="#/" className="flex items-center gap-2.5 group">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-line bg-card text-accent">
              <Stethoscope className="h-[18px] w-[18px]" />
            </span>
            <span className="text-[12px] font-bold uppercase tracking-[0.22em] text-fg">
              Surgical Video Intelligence
            </span>
          </a>

          <nav className="hidden md:flex items-center gap-1">
            {NAV.map((n) => (
              <button
                key={n.id}
                onClick={() => goTo(n.id)}
                className={`px-3 py-1.5 text-[12px] font-medium rounded-md transition-colors ${
                  active === n.id
                    ? 'text-accent bg-accent-soft'
                    : 'text-fg2 hover:text-fg hover:bg-hover'
                }`}
              >
                {n.label}
              </button>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <ThemeToggle />
            <a
              href="#/console"
              className="flex items-center gap-1.5 whitespace-nowrap rounded-lg bg-accent px-4 py-2 text-[12px] font-semibold text-on-accent shadow-[0_2px_14px_rgba(2,132,199,0.28)] transition-colors hover:bg-accent-hover"
            >
              Open console
              <ArrowRight className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>
      </header>

      {/* ---------------------------------------------------------------- */}
      {/* HERO                                                             */}
      {/* ---------------------------------------------------------------- */}
      <div className="glow-field overflow-hidden">
        <div className="relative max-w-[1120px] mx-auto px-6 pt-14 pb-14 sm:pt-20 sm:pb-20">
          <div className="grid items-center gap-12 lg:grid-cols-[1.15fr_0.85fr]">
          <div>
          <Reveal>
            <div className="inline-flex flex-wrap items-center gap-x-2 gap-y-1 rounded-full border border-accent/30 bg-accent-soft px-3.5 py-1.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-accent">
              <span className="h-1.5 w-1.5 rounded-full bg-accent" />
              <span>SurgVU 2024</span>
              <span>·</span>
              <span>arXiv:2501.09209</span>
              <span>·</span>
              <span>inference on this machine</span>
            </div>
          </Reveal>

          <Reveal delay={60}>
            <h1 className="mt-6 text-[38px] leading-[1.06] sm:text-[60px] font-extrabold tracking-tight text-fg">
              Every claim on screen says{' '}
              <span className="text-accent">
                where it came from.
              </span>
            </h1>
          </Reveal>

          <Reveal delay={120}>
            <p className="mt-6 max-w-xl text-[16px] sm:text-[17px] leading-relaxed text-fg2">
              This platform plays real robotic surgical video from the{' '}
              <span className="text-fg">Surgical Visual Understanding</span> release, shows
              what that release <em className="text-fg2 not-italic font-medium">records</em>{' '}
              for every moment of it, and — separately, never mixed in — what a model{' '}
              <em className="text-fg2 not-italic font-medium">predicts</em> it can see.
              Detection runs in the browser: frames never leave the page.
            </p>
          </Reveal>

          <Reveal delay={180}>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <a
                href="#/console"
                className="group inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 shadow-[0_2px_14px_rgba(2,132,199,0.28)] text-[13px] font-semibold text-on-accent transition-colors hover:bg-accent-hover"
              >
                Open console
                <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5" />
              </a>
              <a
                href="https://arxiv.org/abs/2501.09209"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-lg border border-line bg-card px-5 py-2.5 text-[13px] font-semibold text-fg transition-colors hover:border-line hover:bg-hover"
              >
                <FileText className="w-4 h-4 text-fg2" />
                The dataset paper
                <ArrowUpRight className="w-3.5 h-3.5 text-fg3" />
              </a>
            </div>
          </Reveal>
          </div>

          <Reveal delay={120}>
            <figure>
              <div className="relative overflow-hidden rounded-lg border border-line shadow-card">
                <img
                  src={`${import.meta.env.BASE_URL}hero.jpg`}
                  alt="Endoscopic view from a robotic surgery recording in the SurgVU 2024 release"
                  className="block w-full"
                  width={960}
                  height={540}
                />
                <figcaption className="absolute bottom-3 left-3 rounded-md bg-black/60 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-white backdrop-blur">
                  SurgVU 2024 · case 001
                </figcaption>
              </div>
              <p className="mt-2 text-right text-[11px] text-fg3">Frame from a bundled excerpt</p>
            </figure>
          </Reveal>
          </div>

          <Reveal delay={240}>
            <div className="mt-14 grid grid-cols-2 lg:grid-cols-4 gap-3">
              <Stat value="840" unit="h" label="Video in the release" note="155 training sessions, 280 clips" />
              <Stat value="18" unit="M" label="Labelled frames" note="60 fps, 1280 × 720" />
              <Stat
                value={BUNDLED.hours.toFixed(1)}
                unit="h"
                label="Bundled in this build"
                note={`${BUNDLED.cases} cases · ${BUNDLED.parts} parts, fully labelled`}
              />
              <Stat value="~35" unit="ms" label="Per frame, in-browser" note="≈22 fps, WASM, 4 threads" />
            </div>
          </Reveal>
        </div>
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* WHY                                                              */}
      {/* ---------------------------------------------------------------- */}
      <Section id="why" className="border-t border-line">
        <SectionHead
          index="01"
          kicker="The problem"
          title="Surgical video is abundant. Trustworthy readings of it are not."
          icon={<AlertTriangle className="w-3.5 h-3.5" />}
          blurb="Robotic surgery produces recorded video and machine-harvested labels as a matter of course. Four properties of that data shaped every decision in this interface."
        />

        <div className="grid gap-4 md:grid-cols-2">
          {[
            {
              icon: <Database className="w-4 h-4" />,
              title: 'The labels are a log, not a look',
              body: 'Tool presence comes from the robot’s instrument installation record, not from vision. An instrument counts as present the whole time it is installed — including while it is occluded or entirely off-screen. The paper says so plainly, and an interface built on these labels has to say so too.',
            },
            {
              icon: <ShieldAlert className="w-4 h-4" />,
              title: 'A prediction that looks like a fact',
              body: 'A recorded annotation and a model output on the same frame are two different kinds of statement. Drawn the same way, the weaker one quietly inherits the authority of the stronger. That single failure is what the whole visual language here is built to prevent.',
            },
            {
              icon: <Timer className="w-4 h-4" />,
              title: 'Per-frame cloud calls do not survive a case',
              body: 'Analysing every frame through a hosted model meant one network round trip and one billed request per frame: the fastest available setting was a frame every two seconds, and twenty frames returned RESOURCE_EXHAUSTED. “Analyse less often” is not advice you can take during an operation.',
            },
            {
              icon: <Layers className="w-4 h-4" />,
              title: '840 hours nobody can watch',
              body: 'At 60 frames per second the release holds roughly 18 million labelled images. Finding the four minutes where a stapler sits on arm 3 during suturing is an indexing problem before it is ever a computer-vision problem.',
            },
          ].map((c, i) => (
            <Reveal key={c.title} delay={i * 60}>
              <Card className="h-full">
                <div className="flex items-center gap-2.5 text-accent">
                  <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-accent/40 bg-accent-soft">
                    {c.icon}
                  </span>
                  <h3 className="text-[14px] font-semibold text-fg">{c.title}</h3>
                </div>
                <p className="mt-3 text-[13px] leading-relaxed text-fg2">{c.body}</p>
              </Card>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* ---------------------------------------------------------------- */}
      {/* DATASET                                                          */}
      {/* ---------------------------------------------------------------- */}
      <Section id="dataset" className="border-t border-line bg-page">
        <SectionHead
          index="02"
          kicker="The dataset"
          title="Surgical Visual Understanding (SurgVU)"
          icon={<Database className="w-3.5 h-3.5" />}
          blurb={
            <>
              Released by Intuitive Surgical and used for the SurgVU challenges hosted at MICCAI’s
              EndoVis. The video was captured at robotic surgery training sessions where trainee and
              expert surgeons perform standardised steps on porcine tissue with a da Vinci system;
              each frame is the surgeon’s own console view, from one channel of the endoscope.
              Instrument presence is harvested automatically from the robot. The eight surgical
              steps were annotated by clinical experts.
            </>
          }
        />

        <div className="grid gap-4 lg:grid-cols-[1.05fr_1fr]">
          <Reveal>
            <Card className="h-full">
              <h3 className="text-[11px] font-mono uppercase tracking-[0.16em] text-fg3">
                Release at a glance
              </h3>
              <div className="mt-3">
                <Spec k="Video clips" v="280, from 155 training sessions" />
                <Spec k="Duration" v="840+ hours" />
                <Spec k="Frame rate / resolution" v="60 fps · 1280 × 720" />
                <Spec k="Labelled images" v="≈ 18 million" />
                <Spec k="Instrument classes" v="12 (8 in the detection validation set)" />
                <Spec k="Surgical steps" v="8, plus free-text step descriptions" />
                <Spec k="Also released" v="Q&A pairs for surgical VQA" />
                <Spec
                  k="Label format"
                  v={<span className="text-[11px]">install (part, time) → uninstall (part, time), arm, name</span>}
                />
              </div>
            </Card>
          </Reveal>

          <Reveal delay={80}>
            <Card className="h-full border-warn-line bg-warn-soft">
              <div className="flex items-center gap-2 text-warn">
                <Quote className="w-4 h-4" />
                <h3 className="text-[11px] font-mono uppercase tracking-[0.16em]">
                  From the paper
                </h3>
              </div>
              <blockquote className="mt-3 border-l-2 border-warn-line pl-4 text-[14px] leading-relaxed text-fg2">
                “At times surgical tools might be obscured or otherwise temporarily not visible
                despite being installed. Consequently, the tool labels can be considered noisy.”
              </blockquote>
              <p className="mt-4 text-[13px] leading-relaxed text-fg2">
                This one sentence is the reason recorded presence is <strong className="font-semibold text-fg">listed
                rather than boxed</strong> in the console. The release states that a stapler is
                mounted on arm 3 — not where it is on screen, and not that it is on screen at all.
                “Recorded but not visible” is normal, and is not a model error.
              </p>
              <p className="mt-3 text-[13px] leading-relaxed text-fg2">
                Presence is also genuinely multi-label: up to four arms carry an instrument at once,
                which is why the ribbon in the console runs one track per arm (USM1–USM4).
              </p>
            </Card>
          </Reveal>
        </div>

        <Reveal className="mt-4">
          <Card>
            <h3 className="text-[11px] font-mono uppercase tracking-[0.16em] text-fg3">
              Vocabulary — 12 instruments
            </h3>
            <div className="mt-3 flex flex-wrap gap-2">
              {TOOL_CLASSES.map((t) => (
                <Chip key={t}>{toolDisplay(t)}</Chip>
              ))}
            </div>
            <h3 className="mt-6 text-[11px] font-mono uppercase tracking-[0.16em] text-fg3">
              Vocabulary — 8 surgical steps
            </h3>
            <div className="mt-3 flex flex-wrap gap-2">
              {TASK_CLASSES.map((t) => (
                <Chip key={t} tone="cyan">
                  {taskDisplay(t)}
                </Chip>
              ))}
            </div>
          </Card>
        </Reveal>

        <Reveal className="mt-4" delay={60}>
          <Card>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="text-[11px] font-mono uppercase tracking-[0.16em] text-fg3">
                What ships in this build
              </h3>
              <span className="rounded-md border border-ok-line bg-ok-soft px-2 py-0.5 text-[10px] font-mono text-ok">
                generated from the release, verified against a brute-force scan
              </span>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
              {[
                { v: String(BUNDLED.cases), l: 'cases' },
                { v: String(BUNDLED.parts), l: 'video parts' },
                { v: BUNDLED.hours.toFixed(1), l: 'hours of video' },
                { v: String(BUNDLED.toolIntervals), l: 'tool intervals' },
                { v: String(BUNDLED.taskIntervals), l: 'step intervals' },
                { v: String(BUNDLED.toolClasses), l: 'classes present' },
              ].map((s) => (
                <div key={s.l} className="rounded-lg border border-line bg-card px-3 py-2.5">
                  <div className="font-mono text-lg font-bold text-fg">{s.v}</div>
                  <div className="text-[11px] text-fg3">{s.l}</div>
                </div>
              ))}
            </div>

            <p className="mt-4 text-[13px] leading-relaxed text-fg2">
              The annotations are tiny and are bundled. The video is{' '}
              {BUNDLED.gigabytes.toFixed(1)} GB and is not — the console asks you to attach the
              local <code className="rounded bg-inset px-1 py-0.5 font-mono text-[11px] text-fg2">surgvu24_videos_only</code>{' '}
              folder, then reads files straight off disk through{' '}
              <code className="rounded bg-inset px-1 py-0.5 font-mono text-[11px] text-fg2">URL.createObjectURL</code>.
              Nothing is uploaded, and a five-hour part seeks as fast as the disk can serve it.
              Files are matched to cases <strong className="font-semibold text-fg">by filename</strong>, never by
              the order they were picked: playing case 3 against case 2’s labels would produce
              confident, precisely-timed, entirely wrong ground truth, so an unrecognised filename is
              rejected rather than guessed at.
            </p>
          </Card>
        </Reveal>
      </Section>

      {/* ---------------------------------------------------------------- */}
      {/* MODELS                                                           */}
      {/* ---------------------------------------------------------------- */}
      <Section id="models" className="border-t border-line">
        <SectionHead
          index="03"
          kicker="The models"
          title="Three checkpoints, running inside the browser tab"
          icon={<Cpu className="w-3.5 h-3.5" />}
          blurb="Trained on the SurgVU labels, exported to ONNX and executed by onnxruntime-web in a worker. This replaced one hosted-model call per frame — a design whose latency and quota faults no amount of tuning fixes. Each checkpoint prints its own provenance note in the console, verbatim, next to its numbers."
        />

        <div className="grid gap-4 lg:grid-cols-3">
          {[
            {
              name: 'Surgical step recognition',
              file: 'task.onnx',
              arch: 'ConvNeXtV2-Atto · 3.4M params',
              input: '176 px · softmax over 8 classes',
              metric: 'balanced accuracy 0.776 · accuracy 0.844',
              metricNote: 'held out over 20 unseen cases, split by case',
              tone: 'emerald' as const,
              note: 'Answers a question whose answer changes over minutes, so it runs every twelfth frame rather than every frame.',
            },
            {
              name: 'Instrument presence',
              file: 'tool.onnx',
              arch: 'ConvNeXtV2-Atto · 3.4M params',
              input: '224 px · sigmoid, multi-label over 14 classes',
              metric: 'validation mAP 0.971',
              metricNote: 'fitted to five clips of the public cat1 subset, validated on two held out',
              tone: 'amber' as const,
              note: 'The high mAP is real and small. Six of the fourteen classes never occur in that data and cannot be predicted at all.',
            },
            {
              name: 'Instrument localisation',
              file: 'detection.onnx',
              arch: 'YOLO11n · 2.6M params',
              input: '448 px · boxes over 14 classes',
              metric: 'no published validation metric',
              metricNote: 'measured over 60 real frames: ≈1.4 boxes/frame at mean score 0.63',
              tone: 'amber' as const,
              note: 'It under-detects rather than over-detects. Read its boxes as a demonstration of the pipeline, not as a validated detector.',
            },
          ].map((m, i) => (
            <Reveal key={m.file} delay={i * 70}>
              <Card className="h-full flex flex-col">
                <div className="flex items-start justify-between gap-3">
                  <h3 className="text-[14px] font-semibold text-fg">{m.name}</h3>
                  <code className="shrink-0 rounded bg-inset px-1.5 py-0.5 font-mono text-[10px] text-fg2">
                    {m.file}
                  </code>
                </div>
                <p className="mt-2 font-mono text-[11px] text-fg2">{m.arch}</p>
                <p className="font-mono text-[11px] text-fg3">{m.input}</p>

                <div
                  className={`mt-4 rounded-lg border px-3 py-2.5 ${
                    m.tone === 'emerald'
                      ? 'border-ok-line bg-ok-soft'
                      : 'border-warn-line bg-warn-soft'
                  }`}
                >
                  <div
                    className={`font-mono text-[12px] font-semibold ${
                      m.tone === 'emerald' ? 'text-ok' : 'text-warn'
                    }`}
                  >
                    {m.metric}
                  </div>
                  <div className="mt-0.5 text-[11px] leading-snug text-fg2">{m.metricNote}</div>
                </div>

                <p className="mt-4 text-[13px] leading-relaxed text-fg2">{m.note}</p>
              </Card>
            </Reveal>
          ))}
        </div>

        <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_1.1fr]">
          <Reveal>
            <Card className="h-full">
              <div className="flex items-center gap-2 text-accent">
                <Gauge className="w-4 h-4" />
                <h3 className="text-[11px] font-mono uppercase tracking-[0.16em]">
                  Measured end to end, in the browser
                </h3>
              </div>
              <p className="mt-2 text-[12px] leading-relaxed text-fg3">
                On a video frame, including preprocessing and box decoding. The console toolbar
                prints the live figure and which backend produced it, so the number is checkable
                rather than claimed.
              </p>
              <table className="mt-4 w-full text-left">
                <thead>
                  <tr className="text-[10px] font-mono uppercase tracking-wider text-fg3">
                    <th className="pb-2 font-normal">Backend</th>
                    <th className="pb-2 font-normal text-right">Per frame</th>
                    <th className="pb-2 font-normal text-right">Rate</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-[12px]">
                  <tr className="border-t border-line">
                    <td className="py-2.5 text-fg2">WASM, 4 threads</td>
                    <td className="py-2.5 text-right font-semibold text-accent">~35 ms</td>
                    <td className="py-2.5 text-right text-fg2">~22 fps</td>
                  </tr>
                  <tr className="border-t border-line">
                    <td className="py-2.5 text-fg2">WASM, single thread</td>
                    <td className="py-2.5 text-right text-fg2">~98 ms</td>
                    <td className="py-2.5 text-right text-fg2">~10 fps</td>
                  </tr>
                </tbody>
              </table>
              <p className="mt-3 text-[11px] leading-snug text-fg3">
                The difference between the two rows is cross-origin isolation: the server sends
                COOP and COEP so onnxruntime-web can use SharedArrayBuffer. It is best-effort — an
                iframe is only isolated if its embedder is — and the toolbar says which happened.
              </p>
            </Card>
          </Reveal>

          <Reveal delay={80}>
            <Card className="h-full">
              <div className="flex items-center gap-2 text-accent">
                <Boxes className="w-4 h-4" />
                <h3 className="text-[11px] font-mono uppercase tracking-[0.16em]">
                  Why it is fast enough
                </h3>
              </div>
              <ul className="mt-3 space-y-3">
                {[
                  ['Small models.', 'Two 3.4M-parameter classifiers and a 2.6M-parameter detector — chosen for the frame budget, not the leaderboard.'],
                  ['A worker.', 'The main thread is playing video. A 20 ms inference there would drop frames.'],
                  ['One frame in flight, newest wins.', 'Frames are dropped rather than queued. A slow machine analyses fewer frames; it never falls behind.'],
                  ['Cadence split.', 'The detector runs on every frame. The step and presence classifiers run every twelfth.'],
                ].map(([lead, rest]) => (
                  <li key={lead} className="flex gap-3 text-[13px] leading-relaxed text-fg2">
                    <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-accent-soft" />
                    <span>
                      <strong className="font-semibold text-fg">{lead}</strong> {rest}
                    </span>
                  </li>
                ))}
              </ul>
              <p className="mt-4 border-t border-line pt-3 text-[12px] leading-relaxed text-fg3">
                No hosted model is used at all. The assistant answers from the release's own
                annotations and from the bundled knowledge base, both in this page; voice runs
                through the browser's own speech APIs. A visitor who wants generated prose can
                add their own Gemini key under Settings — their quota, their choice, and factual
                lookups stay on the grounded path regardless.
              </p>
            </Card>
          </Reveal>
        </div>
      </Section>

      {/* ---------------------------------------------------------------- */}
      {/* PIPELINE                                                         */}
      {/* ---------------------------------------------------------------- */}
      <Section id="pipeline" className="border-t border-line bg-page">
        <SectionHead
          index="04"
          kicker="The pipeline"
          title="Two lanes, and they never merge"
          icon={<Workflow className="w-3.5 h-3.5" />}
          blurb="One lane runs on every frame, the other when you ask a question, and neither has a network call in it by default. Keeping them separate is what makes the provenance rule below enforceable rather than decorative. A visitor who adds their own API key gets a third path on top of Lane B — never instead of it."
        />

        <Reveal>
          <div className="rounded-xl border border-line bg-card p-5 sm:p-6">
            <div className="flex items-center gap-2">
              <span className="rounded-md border border-accent/40 bg-accent-soft px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-accent">
                Lane A — every frame · on device
              </span>
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-5">
              {[
                ['Frame', 'grabbed from the playing video element'],
                ['Preprocess', 'letterbox, normalise, to tensor — in a worker'],
                ['ONNX Runtime Web', 'WASM SIMD, 4 threads where isolation allows'],
                ['Decode + NMS', 'boxes, presence vector, step softmax'],
                ['Overlay', 'drawn straight onto the frame bus, bypassing app state'],
              ].map((s, i) => (
                <div key={s[0]} className="relative rounded-lg border border-line bg-card p-3">
                  <div className="font-mono text-[10px] text-fg3">
                    {String(i + 1).padStart(2, '0')}
                  </div>
                  <div className="mt-1 text-[13px] font-semibold text-fg">{s[0]}</div>
                  <div className="mt-1 text-[11px] leading-snug text-fg3">{s[1]}</div>
                  {i < 4 && (
                    <ArrowRight className="absolute -right-[15px] top-1/2 hidden h-3.5 w-3.5 -translate-y-1/2 text-fg2 md:block" />
                  )}
                </div>
              ))}
            </div>

            <div className="mt-7 flex items-center gap-2">
              <span className="rounded-md border border-line bg-inset px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-fg2">
                Lane B — on request · in this page
              </span>
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              {[
                ['Ask', "typed, or dictated through the browser's own recogniser — no audio is sent to any API"],
                ['Ground', 'the question is classified, then answered from the installation log, the task intervals and the timeline — a lookup, not a generation'],
                ['Answer', 'badged with the path that produced it; if nothing matched, it says so rather than improvising'],
              ].map((s, i) => (
                <div key={s[0]} className="relative rounded-lg border border-line bg-card p-3">
                  <div className="font-mono text-[10px] text-fg3">
                    {String(i + 1).padStart(2, '0')}
                  </div>
                  <div className="mt-1 text-[13px] font-semibold text-fg">{s[0]}</div>
                  <div className="mt-1 text-[11px] leading-snug text-fg3">{s[1]}</div>
                  {i < 2 && (
                    <ArrowRight className="absolute -right-[15px] top-1/2 hidden h-3.5 w-3.5 -translate-y-1/2 text-fg2 md:block" />
                  )}
                </div>
              ))}
            </div>
          </div>
        </Reveal>
      </Section>

      {/* ---------------------------------------------------------------- */}
      {/* PROVENANCE                                                       */}
      {/* ---------------------------------------------------------------- */}
      <Section id="provenance" className="border-t border-line">
        <SectionHead
          index="05"
          kicker="The design rule"
          title="Recorded and predicted never look alike"
          icon={<ShieldCheck className="w-3.5 h-3.5" />}
          blurb="A prediction with no percentage would be indistinguishable from a recorded fact, so one is always shown. When a model call fails, nothing is substituted for it: the predictions panel stays empty and the banner says why."
        />

        <div className="grid gap-4 lg:grid-cols-2">
          <Reveal>
            <Card className="h-full border-ok-line">
              <div className="flex items-center gap-2 text-ok">
                <ShieldCheck className="w-4 h-4" />
                <h3 className="text-[12px] font-mono uppercase tracking-[0.16em]">Recorded</h3>
              </div>
              <p className="mt-3 text-[13px] leading-relaxed text-fg2">
                The release’s own annotations — the robot’s instrument installation log and the
                expert step labels. A fact about what was mounted, not about what is visible.
              </p>
              <div className="mt-4 rounded-lg border border-ok-line bg-ok-soft p-3">
                <div className="text-[10px] font-mono uppercase tracking-wider text-ok">
                  Installed at 01:42:07
                </div>
                <ul className="mt-2 space-y-1.5">
                  {[
                    ['USM1', 'Monopolar curved scissors'],
                    ['USM2', 'Cadiere forceps'],
                    ['USM3', 'Prograsp forceps'],
                  ].map(([arm, tool]) => (
                    <li key={arm} className="flex items-center gap-2 text-[12px]">
                      <span className="h-1.5 w-1.5 rounded-full bg-ok" />
                      <span className="font-mono text-[10px] text-ok">{arm}</span>
                      <span className="text-fg">{tool}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <p className="mt-3 text-[12px] leading-relaxed text-fg3">
                Solid green, <strong className="text-fg2">listed not boxed</strong>, and{' '}
                <strong className="text-fg2">no confidence</strong> — because none applies.
              </p>
            </Card>
          </Reveal>

          <Reveal delay={80}>
            <Card className="h-full border-warn-line">
              <div className="flex items-center gap-2 text-warn">
                <Eye className="w-4 h-4" />
                <h3 className="text-[12px] font-mono uppercase tracking-[0.16em]">Predicted</h3>
              </div>
              <p className="mt-3 text-[13px] leading-relaxed text-fg2">
                A model, at runtime, on this frame. An estimate about where something is — the
                question the recorded labels cannot answer.
              </p>
              <div className="mt-4 relative aspect-[16/9] overflow-hidden rounded-lg border border-line bg-[radial-gradient(120%_100%_at_30%_20%,#3f1d1d,#170d0d_60%,#0b0708)]">
                <div className="absolute inset-0 opacity-30 [background-image:radial-gradient(circle_at_70%_60%,rgba(255,180,150,0.25),transparent_45%)]" />
                <div className="absolute left-[16%] top-[28%] h-[44%] w-[38%] rounded-[3px] border-2 border-dashed border-warn-line">
                  <span className="absolute -top-6 left-0 whitespace-nowrap rounded bg-warn px-1.5 py-0.5 font-mono text-[10px] font-bold text-on-accent">
                    monopolar curved scissors · 63%
                  </span>
                </div>
                <div className="absolute right-[14%] bottom-[20%] h-[30%] w-[26%] rounded-[3px] border-2 border-dashed border-warn-line">
                  <span className="absolute -top-6 left-0 whitespace-nowrap rounded bg-warn-soft px-1.5 py-0.5 font-mono text-[10px] font-bold text-black">
                    cadiere forceps · 41%
                  </span>
                </div>
              </div>
              <p className="mt-3 text-[12px] leading-relaxed text-fg3">
                Dashed amber box over the video,{' '}
                <strong className="text-fg2">always with a percentage</strong>.
              </p>
            </Card>
          </Reveal>
        </div>

        <Reveal className="mt-4" delay={40}>
          <Card>
            <div className="flex items-center gap-2 text-fg2">
              <AlertTriangle className="w-4 h-4 text-warn" />
              <h3 className="text-[11px] font-mono uppercase tracking-[0.16em]">
                What this build is not
              </h3>
            </div>
            <ul className="mt-3 grid gap-2.5 md:grid-cols-2">
              {[
                'Not a medical device. Surgical education and computer-vision research only.',
                'The footage is porcine tissue in a controlled training environment, not clinical cases.',
                'Presence labels are installation logs, so “recorded but not visible” is expected behaviour, not a miss.',
                'The presence and detection checkpoints were fitted to a handful of clips; six of their fourteen classes never occur in that data.',
                'Recorded labels can name classes no checkpoint has a head for — a suction irrigator, for one — and those are marked rather than scored.',
                'No prediction is ever shown without a confidence, and no failed call is ever backfilled with a guess.',
              ].map((t) => (
                <li key={t} className="flex gap-2.5 text-[13px] leading-relaxed text-fg2">
                  <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-inset" />
                  <span>{t}</span>
                </li>
              ))}
            </ul>
          </Card>
        </Reveal>
      </Section>

      {/* ---------------------------------------------------------------- */}
      {/* AUTHOR                                                           */}
      {/* ---------------------------------------------------------------- */}
      <Section id="author" className="border-t border-line bg-page">
        <SectionHead
          index="06"
          kicker="The author"
          title="Built by Kim Nguyen"
          icon={<GraduationCap className="w-3.5 h-3.5" />}
        />

        <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
          <Reveal>
            <Card className="h-full">
              <div className="flex items-start gap-4">
                <img
                  src={`${import.meta.env.BASE_URL}author.jpg`}
                  alt="Kim Nguyen"
                  width={96}
                  height={96}
                  className="h-24 w-24 shrink-0 rounded-lg border border-line object-cover"
                />
                <div>
                  <h3 className="text-[18px] font-bold tracking-tight text-fg">Kim Nguyen</h3>
                  <p className="mt-1 text-[13px] font-medium text-accent">
                    MSc student, Smart Medicine &amp; Health Informatics
                  </p>
                  <p className="text-[13px] text-fg2">National Taiwan University</p>
                </div>
              </div>

              <div className="mt-5 space-y-3 text-[13px] leading-relaxed text-fg2">
                <p>
                  I came to data science from a non-traditional background in international business
                  and taught myself the field, because what I wanted to build sat at the
                  intersection of healthcare and data. My interest is in clinical AI that is
                  explainable and honest about its own limits — systems that are not only
                  intelligent, but meaningful in a real clinical setting.
                </p>
                <p>
                  This project is that interest applied to surgical video. The hard part was never
                  getting a model to draw a box; it was making sure a viewer can always tell a
                  drawn box apart from a recorded fact, and that every number on screen carries the
                  provenance of the data it came from.
                </p>
              </div>

              <div className="mt-6 flex flex-wrap gap-2">
                {[
                  'Surgical data science',
                  'Explainable clinical AI',
                  'Biomedical image analysis',
                  'PyTorch · ONNX',
                  'Health informatics',
                ].map((t) => (
                  <Chip key={t} tone="cyan">
                    {t}
                  </Chip>
                ))}
              </div>
            </Card>
          </Reveal>

          <Reveal delay={80}>
            <div className="flex h-full flex-col gap-3">
              <a
                href="https://kimnguyen2002.github.io/Portfolio/"
                target="_blank"
                rel="noreferrer"
                className="group flex-1 rounded-xl border border-line bg-card p-5 transition-colors hover:border-accent/40 hover:bg-hover"
              >
                <div className="flex items-center justify-between">
                  <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-line bg-card text-fg2 group-hover:text-accent">
                    <Globe className="h-4 w-4" />
                  </span>
                  <ArrowUpRight className="h-4 w-4 text-fg3 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-accent" />
                </div>
                <div className="mt-4 text-[14px] font-semibold text-fg">Portfolio</div>
                <div className="font-mono text-[11px] text-fg3">
                  kimnguyen2002.github.io/Portfolio
                </div>
                <p className="mt-2 text-[12px] leading-snug text-fg3">
                  Projects across clinical prediction, biomedical imaging and health informatics.
                </p>
              </a>

              <a
                href="https://github.com/kimnguyen2002"
                target="_blank"
                rel="noreferrer"
                className="group flex-1 rounded-xl border border-line bg-card p-5 transition-colors hover:border-accent/40 hover:bg-hover"
              >
                <div className="flex items-center justify-between">
                  <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-line bg-card text-fg2 group-hover:text-accent">
                    <Github className="h-4 w-4" />
                  </span>
                  <ArrowUpRight className="h-4 w-4 text-fg3 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-accent" />
                </div>
                <div className="mt-4 text-[14px] font-semibold text-fg">GitHub</div>
                <div className="font-mono text-[11px] text-fg3">github.com/kimnguyen2002</div>
                <p className="mt-2 text-[12px] leading-snug text-fg3">
                  Source for this project and the rest of the work.
                </p>
              </a>
            </div>
          </Reveal>
        </div>
      </Section>

      {/* ---------------------------------------------------------------- */}
      {/* REFERENCES                                                       */}
      {/* ---------------------------------------------------------------- */}
      <Section id="references" className="border-t border-line">
        <Reveal>
          <div className="flex items-center gap-3 text-[11px] font-mono uppercase tracking-[0.18em] text-fg3">
            <FileText className="h-3.5 w-3.5" />
            <span>References &amp; data</span>
          </div>
        </Reveal>

        <Reveal className="mt-6" delay={60}>
          <Card>
            <p className="text-[13px] leading-relaxed text-fg2">
              A. Zia, M. Berniker, R. Nespolo, X. Zhang, C. Perreault, Z. Wang, B. Mueller, R.
              Schmidt, K. Bhattacharyya, X. Liu and A. Jarc.{' '}
              <span className="font-semibold text-fg">
                “Surgical Visual Understanding (SurgVU) Dataset.”
              </span>{' '}
              Intuitive Surgical, Inc.{' '}
              <a
                href="https://arxiv.org/abs/2501.09209"
                target="_blank"
                rel="noreferrer"
                className="font-mono text-accent underline decoration-cyan-500/40 underline-offset-2 hover:text-accent"
              >
                arXiv:2501.09209
              </a>
              .
            </p>
            <p className="mt-3 text-[12px] leading-relaxed text-fg3">
              The dataset accompanies the SurgVU challenges hosted each year by Intuitive Surgical
              at MICCAI, as part of the Endoscopic Vision (EndoVis) challenge.
            </p>

            <div className="mt-5 grid gap-2 sm:grid-cols-2">
              {[
                ['Videos', 'surgvu24_videos_only.zip', 'https://storage.googleapis.com/isi-surgvu/surgvu24_videos_only.zip'],
                ['Labels', 'surgvu24_labels_updated_v2.zip', 'https://storage.googleapis.com/isi-surgvu/surgvu24_labels_updated_v2.zip'],
                ['Detection validation set', 'cat1_test_set_public.zip', 'https://storage.googleapis.com/isi-surgvu/cat1_test_set_public.zip'],
                ['VQA sample set', 'SURGVU25_cat_2_sample_set_public.zip', 'https://storage.googleapis.com/isi-surgvu/SURGVU25_cat_2_sample_set_public.zip'],
              ].map(([label, file, href]) => (
                <a
                  key={file}
                  href={href}
                  target="_blank"
                  rel="noreferrer"
                  className="group flex items-center justify-between gap-3 rounded-lg border border-line bg-card px-3 py-2.5 transition-colors hover:border-line"
                >
                  <span>
                    <span className="block text-[12px] font-medium text-fg2">{label}</span>
                    <span className="block font-mono text-[10px] text-fg3">{file}</span>
                  </span>
                  <ArrowUpRight className="h-3.5 w-3.5 shrink-0 text-fg3 group-hover:text-accent" />
                </a>
              ))}
            </div>
          </Card>
        </Reveal>

        <Reveal className="mt-10" delay={100}>
          <div className="relative overflow-hidden rounded-lg border border-line bg-card px-6 py-10 text-center">
            <h3 className="text-[22px] font-bold tracking-tight text-fg sm:text-[26px]">
              See it running on a real case
            </h3>
            <p className="mx-auto mt-3 max-w-xl text-[14px] leading-relaxed text-fg2">
              Attach the SurgVU folder, pick a part, and press analyse. Recorded labels appear as
              soon as the file binds; the on-device models start the moment the video plays.
            </p>
            <a
              href="#/console"
              className="group mt-6 inline-flex items-center gap-2 rounded-lg bg-accent px-6 py-3 text-[13px] font-semibold text-on-accent transition-colors hover:bg-accent-hover"
            >
              Open the console
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </a>
          </div>
        </Reveal>
      </Section>

      <footer className="border-t border-line px-6 py-8">
        <div className="mx-auto flex max-w-[1120px] flex-col items-center gap-2 text-center">
          <p className="text-[11px] leading-relaxed text-fg3">
            Recorded labels from the SurgVU 2024 release · predictions from a model, always shown
            with a confidence · educational and research use only, not a medical device.
          </p>
          <p className="font-mono text-[10px] text-fg2">
            Kim Nguyen · National Taiwan University · Smart Medicine &amp; Health Informatics
          </p>
        </div>
      </footer>
    </div>
  );
}
