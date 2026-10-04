/**
 * The assistant's conversational layer: everything it can say without a
 * language model that is not a lookup into the video.
 *
 * It runs before every other path, and it does two jobs:
 *
 * 1. **Small talk and orientation.** "Hi", "what is this for?", "how do I
 *    start?", "what does a needle driver do?" — questions with a fixed, correct
 *    answer that no visitor should need an API key to get.
 * 2. **Refusal.** Patient-specific, diagnostic, treatment and medication
 *    questions are declined here, *before* retrieval or generation can produce
 *    something fluent and wrong. This is an educational demo trained on porcine
 *    training footage; a confident answer to a clinical question would be the
 *    most harmful thing it could say, so it never tries.
 *
 * Everything here is pattern-matched and written by hand. Nothing is generated.
 */

import type { VideoFacts } from './facts';
import { hms } from './facts';

export type ConversationKind =
  | 'emergency'
  | 'clinical_advice'
  | 'greeting'
  | 'thanks'
  | 'farewell'
  | 'wellbeing'
  | 'about'
  | 'howto'
  | 'author'
  | 'privacy'
  | 'dataset'
  | 'models'
  | 'accuracy'
  | 'definition';

export interface ConversationalAnswer {
  text: string;
  kind: ConversationKind;
  /** True when the assistant is declining to answer. */
  refusal: boolean;
}

// ---------------------------------------------------------------------------
// Glossary: what each instrument and task in the release is.
// ---------------------------------------------------------------------------

interface GlossaryEntry {
  name: string;
  /** Lower-case phrases that identify the entry in a question. */
  aliases: string[];
  text: string;
}

const INSTRUMENTS: GlossaryEntry[] = [
  {
    name: 'Needle driver',
    aliases: ['needle driver', 'needle holder', 'suturecut'],
    text:
      'holds and drives a curved suture needle through tissue, and is used for suturing and knot tying. The "SutureCut" version recorded in this release also has a small blade to cut the suture.',
  },
  {
    name: 'Cadiere forceps',
    aliases: ['cadiere'],
    text:
      'a grasper with long, fenestrated (windowed) jaws, designed to hold and retract tissue gently without crushing it.',
  },
  {
    name: 'Prograsp forceps',
    aliases: ['prograsp', 'pro grasp'],
    text:
      'a fenestrated grasper with a firmer grip than Cadiere forceps, used to hold and retract heavier tissue.',
  },
  {
    name: 'Monopolar curved scissors',
    aliases: ['monopolar curved scissors', 'monopolar scissors', 'curved scissors', 'hot shears'],
    text:
      'curved scissors that cut tissue mechanically and can also pass monopolar electrical energy to cut and coagulate (cautery) during dissection.',
  },
  {
    name: 'Bipolar forceps',
    aliases: ['bipolar forceps', 'maryland bipolar', 'fenestrated bipolar'],
    text:
      'a grasper that passes bipolar energy between its two jaws, so it can hold tissue and coagulate small vessels between the jaws.',
  },
  {
    name: 'Force bipolar',
    aliases: ['force bipolar'],
    text: 'a bipolar grasper with a stronger grip, used both to hold tissue and to coagulate it.',
  },
  {
    name: 'Bipolar dissector',
    aliases: ['bipolar dissector'],
    text: 'a fine-tipped bipolar instrument for delicate dissection with coagulation.',
  },
  {
    name: 'Vessel sealer',
    aliases: ['vessel sealer', 'vessel sealing'],
    text:
      'uses bipolar energy and pressure to seal vessels and tissue, then divides the sealed tissue with a built-in blade.',
  },
  {
    name: 'Permanent cautery hook/spatula',
    aliases: ['cautery hook', 'cautery spatula', 'permanent cautery', 'hook', 'spatula'],
    text:
      'a monopolar electrode shaped as a hook or a spatula, used to dissect and coagulate tissue with electrical energy.',
  },
  {
    name: 'Clip applier',
    aliases: ['clip applier', 'clip applicator', 'clipper'],
    text: 'places small clips across a vessel or duct to close it off before it is divided.',
  },
  {
    name: 'Stapler',
    aliases: ['stapler', 'staple'],
    text:
      'fires rows of staples across tissue and usually cuts between them, so the tissue is divided and both ends are sealed in one step.',
  },
  {
    name: 'Tip-up fenestrated grasper',
    aliases: ['tip-up', 'tip up', 'fenestrated grasper'],
    text: 'a grasper whose jaw tips curve upward, used to lift and retract tissue gently.',
  },
  {
    name: 'Grasping retractor',
    aliases: ['grasping retractor', 'retractor'],
    text: 'holds organs or tissue out of the way so the working area stays exposed.',
  },
  {
    name: 'Suction irrigator',
    aliases: ['suction irrigator', 'suction', 'irrigator', 'irrigation'],
    text:
      'washes the field with fluid and suctions away fluid, blood and smoke. The release records it, but none of this app\'s models were trained to recognise it.',
  },
];

const TASKS: GlossaryEntry[] = [
  {
    name: 'Suturing',
    aliases: ['suturing', 'suture', 'stitching', 'knot tying'],
    text:
      'placing stitches with a needle and thread and tying knots. In SurgVU it is a standardised training exercise.',
  },
  {
    name: 'Uterine horn',
    aliases: ['uterine horn'],
    text:
      'a training exercise that dissects and manipulates the uterine horn of the porcine model, practising careful tissue handling.',
  },
  {
    name: 'Suspensory ligaments',
    aliases: ['suspensory ligament'],
    text: 'a training exercise that dissects the suspensory ligaments, practising controlled dissection.',
  },
  {
    name: 'Rectal artery/vein manipulation',
    aliases: ['rectal artery', 'rectal vein', 'rectal artery/vein'],
    text:
      'a training exercise that isolates and handles the rectal artery and vein, practising work around vessels.',
  },
  {
    name: 'General skills application',
    aliases: ['skills application', 'general skills'],
    text: 'segments where general robotic handling skills are practised rather than one named exercise.',
  },
  {
    name: 'Range of motion',
    aliases: ['range of motion'],
    text: 'an exercise that moves the instruments through their range to practise control of the wristed tips.',
  },
  {
    name: 'Retraction and collision avoidance',
    aliases: ['collision avoidance', 'retraction and collision', 'retraction'],
    text: 'an exercise in retracting tissue while keeping the robotic arms and instruments from colliding.',
  },
];

// ---------------------------------------------------------------------------
// Patterns. Order matters: safety first, then orientation, then definitions.
// ---------------------------------------------------------------------------

const EMERGENCY =
  /\b(emergency|can'?t breathe|cannot breathe|chest pain|heart attack|stroke|suicid\w*|kill (my|him|her)self|overdos\w*|unconscious|passed out|bleeding (heavily|a lot|won'?t stop))\b/i;

const MEDICATION =
  /\b(dose|doses|dosage|dosing|mg|mcg|milligrams?|prescri\w*|medications?|medicines?|drugs?|antibiotics?|painkillers?|opioids?|anticoagula\w*|blood thinners?|ibuprofen|paracetamol|acetaminophen|aspirin|heparin|warfarin|insulin|anaesthe\w*|anesthe\w*)\b/i;

const PERSONAL =
  /\b(my|our) (patient|mother|mom|mum|father|dad|wife|husband|partner|son|daughter|child|kid|baby|friend|grandm\w*|grandf\w*|surgery|operation|procedure|wound|incision|scar|stitches|symptoms?|pain|results?|scan|biopsy|tumou?r|cancer|diagnosis|condition)\b|\b(i have|i've got|i am having|i'm having|i feel|i've been feeling|i am feeling|i'm feeling)\b.*\b(pain|aches?|bleed\w*|fever|swell\w*|lumps?|symptoms?|infect\w*|nause\w*|dizz\w*|sick|unwell|vomit\w*|rash|cough\w*|hurts?)\b/i;

const DECISION =
  /\b(should i|should we|should my|do i need|is it (safe|normal|dangerous|ok|okay) (for me|for my|if i)|can i (eat|drink|shower|drive|exercise|work|fly|take|stop|start)|diagnos\w*|prognos\w*|survival rate|life expectancy|second opinion|recover(y)? time for|which (surgery|surgeon|hospital|treatment|procedure|operation) (should|is best|is better)|treat(ment)? (for|of) (my|a patient|this patient))\b/i;

const OPERATIVE_INSTRUCTION =
  /\bhow (do|should|can|would) (i|we|you) (perform|operate|do|carry out|undertake)\b.*\b(\w+ectomy|\w+otomy|\w+plasty|surgery|operation|resection|anastomosis|repair|transplant)\b/i;

/** Questions about this video, which other paths answer from the data. */
const ABOUT_THIS_VIDEO =
  /\b(now|currently|right now|on screen|on-screen|installed|in (this|the) (video|frame|clip|case)|here|at this (time|moment|point)|visible)\b/i;

const GREETING =
  /^(hi+|hello+|hey+|hiya|howdy|yo|greetings|good (morning|afternoon|evening|day)|hi there|hello there|hey there|xin chào|chào|ni ?hao|你好)( (assistant|there|everyone|all|friend))?$/i;
const WELLBEING = /^(how are you|how are you doing|how's it going|how is it going|what's up|sup|how do you do)$/i;
const THANKS =
  /^(thanks?|thank you|thank u|thx|ty|cheers|many thanks|great|awesome|cool|nice|perfect|ok|okay|got it|understood|that helps|helpful)( (so much|a lot|very much|again|for (the|your) help))?$/i;
const FAREWELL = /^(bye|goodbye|good bye|see (you|ya)|see you later|later|good night|goodnight)$/i;

const ABOUT =
  /\b(what (is|'s) (this|the) (tool|app|application|site|website|page|platform|project|demo|thing|assistant)|what (is|'s) this( for| about)?$|what (is|'s) (this|it) (used )?for|what (can|do|could) you do|what are you|who are you|what does (this|it) do|what('s| is) the (purpose|point|goal)|how does (this|it|the app|the tool) work|what can i (ask|do)|your (capabilit\w*|features?)|what are your (capabilit\w*|features?)|^help$|^help me$|^\?$)/i;
const HOWTO =
  /\b(how (do|can|should) i (use|start|begin|get started|load|play|open|analy[sz]e|run|try)|get(ting)? started|where (do|should) i (start|begin)|how to (use|start)|guide me|tutorial|first steps?)\b/i;
const AUTHOR = /\b(who (made|built|created|developed|wrote|designed|is behind) (this|it|you|the (app|tool|project))|author|creator|developer of)\b/i;
const PRIVACY =
  /\b(privacy|private|upload\w*|my (data|video|footage)|sent to (a|the) (server|cloud)|stored?|tracking|is (it|this) free|does (it|this) cost|cost money|pay for|api key|need a key)\b/i;
const DATASET = /\b(dataset|surgvu|where (does|do|did) (the|this) (video|footage|data) come from|what (data|footage|videos?) (is|are|was|were))\b/i;
const MODELS =
  /\b(what (models?|ai|neural networks?|networks?) (is|are|do|does)|which models?|what model|onnx|yolo\w*|convnext\w*|how (does|do) (the )?(detection|detector|model|ai|heatmap|cam) work)\b/i;
const ACCURACY = /\b(how accurate|accuracy|how reliable|reliab\w*|can i trust|trustworthy|how good is|performance|precision|recall|error rate)\b/i;

const DEFINITION = /\b(what (is|are|does|do)|what's|define|definition|explain|tell me about|meaning of|purpose of|used for|use of|for what)\b/i;

function normalise(question: string): string {
  return question
    .toLowerCase()
    .replace(/[!?.,;:~]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function findGlossary(q: string): { entry: GlossaryEntry; kind: 'instrument' | 'task' } | null {
  // Longest alias first, so "grasping retractor" wins over "retractor".
  const all = [
    ...INSTRUMENTS.map((entry) => ({ entry, kind: 'instrument' as const })),
    ...TASKS.map((entry) => ({ entry, kind: 'task' as const })),
  ].flatMap((x) => x.entry.aliases.map((alias) => ({ ...x, alias })));
  all.sort((a, b) => b.alias.length - a.alias.length);
  for (const { entry, kind, alias } of all) {
    if (new RegExp(`\\b${alias.replace(/[-/]/g, '[-/ ]?')}`, 'i').test(q)) return { entry, kind };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Replies.
// ---------------------------------------------------------------------------

const DISCLAIMER =
  'This assistant is an educational demo built on porcine training footage, not a medical device.';

const REFUSAL = [
  "I'm sorry, I can't help with that one.",
  '',
  `${DISCLAIMER} It is not trained or validated to give diagnoses, treatment or medication advice, or guidance about a specific patient, and a wrong answer to that kind of question could cause real harm. I won't guess.`,
  '',
  'Please ask a qualified clinician. If it is urgent, contact your local emergency number.',
  '',
  'What I *can* do: explain what is happening in the video, what an instrument or training task is, and general surgical-education topics from the built-in knowledge base.',
].join('\n');

const EMERGENCY_REPLY = [
  '**If someone may be in danger, contact your local emergency number now.**',
  '',
  `I can't help with emergencies or personal medical situations. ${DISCLAIMER}`,
].join('\n');

const CAPABILITIES = [
  '- **"What instruments are installed right now?"** (from the dataset\'s own labels)',
  '- **"What task is this?"** or **"When does suturing happen?"**',
  '- **"Summarise this case"**',
  '- **"What is a needle driver used for?"** (instruments and training tasks)',
  '- **"What anatomy is at risk during this task?"** (educational notes)',
].join('\n');

function videoContext(facts: VideoFacts): string | null {
  const truth = facts.groundTruth;
  if (!truth) return null;
  return `You're watching case ${truth.caseId}, part ${truth.part}, at ${hms(truth.timestamp)} in the original recording.`;
}

function greeting(facts: VideoFacts): string {
  const context = videoContext(facts);
  return [
    'Hi! I can tell you what is happening in this robotic surgery video, using the dataset\'s own labels and the models running in your browser.',
    '',
    context ?? 'Pick a case in the library below the player to load a clip, then ask me about it.',
    '',
    'You could try:',
    CAPABILITIES,
  ].join('\n');
}

const ABOUT_REPLY = [
  '**Surgical Video Intelligence** plays robotic surgery training video from the **SurgVU 2024** dataset and shows two things side by side:',
  '',
  '- **Recorded**: what the dataset\'s annotators and the robot\'s logs say (which instruments are installed, which training task is under way). These are facts and carry no confidence score.',
  '- **Predicted**: what three small AI models, running inside your browser, think they see (instrument boxes, instrument presence, the current task). These always come with a confidence percentage.',
  '',
  'Nothing is uploaded and no API key is needed. I answer from the labels and a built-in knowledge base, and I say so when I don\'t know.',
  '',
  'Things you can ask me:',
  CAPABILITIES,
].join('\n');

const HOWTO_REPLY = [
  'Getting started takes three steps:',
  '',
  '1. **Pick a clip.** In the case library under the player, open a case and click a part with the scissors icon. Each is a two-minute excerpt.',
  '2. **Play and analyse.** Press **Play**, then **Analyse frames (live)**. Yellow dashed boxes are the model\'s predictions. Turn on **Heatmap** to see where the classifier found evidence.',
  '3. **Ask me.** For example "What instruments are installed right now?" or "Summarise this case".',
  '',
  'The **Guide** button in the header walks you through the screen.',
].join('\n');

const AUTHOR_REPLY = [
  'This project was built by **Kim Nguyen**, an MSc student in Smart Medicine & Health Informatics at National Taiwan University.',
  '',
  'The **Overview** page (header button) has more about the project, the dataset and the models, plus links to the portfolio and GitHub.',
].join('\n');

const PRIVACY_REPLY = [
  'Everything runs in your browser:',
  '',
  '- **Video** frames are analysed on your own machine and never uploaded.',
  '- **Answers** like this one come from the dataset labels and a built-in knowledge base, with no server and no cost.',
  '- **Optional**: you can add your *own* Gemini API key under Settings for generated explanations. It is stored only in this browser and sent only to Google.',
].join('\n');

const DATASET_REPLY = [
  'The video comes from **SurgVU 2024** (Surgical Visual Understanding, arXiv:2501.09209), released for the MICCAI EndoVis challenge.',
  '',
  'It was recorded at robotic surgery **training sessions**: surgeons perform standardised exercises on **porcine (pig) tissue** with a da Vinci system. Its labels say which instrument is installed on each robotic arm and which training task is under way.',
  '',
  'This app bundles a two-minute excerpt from each of 9 recording parts, plus the full label set for those cases.',
].join('\n');

const MODELS_REPLY = [
  'Three small models run inside your browser (ONNX Runtime Web, in a background worker):',
  '',
  '- **Instrument localisation**: a YOLO11n detector that draws boxes around instruments.',
  '- **Instrument presence**: a ConvNeXtV2-Atto classifier that says which instruments are in view. It also powers the heatmap, which shows where in the frame its evidence came from.',
  '- **Task recognition**: a second ConvNeXtV2-Atto classifier that predicts the training task.',
  '',
  'The **Overview** page lists their training data and measured numbers.',
].join('\n');

const ACCURACY_REPLY = [
  'Honestly: treat the predictions as a demonstration, not as a validated tool.',
  '',
  '- **Task recognition**: balanced accuracy 0.776 on 20 recordings it never saw. This is the most trustworthy number.',
  '- **Instrument detector**: puts at least one box on about 81% of frames from the bundled clips, and has no published validation metric. It under-detects rather than over-detects.',
  '- **Instrument presence**: a high validation score (mAP 0.971), but on a small dataset, and several instrument classes never appear in it.',
  '',
  'That is why **recorded** labels and **predicted** output are always shown separately, and why every prediction carries its confidence.',
].join('\n');

function definitionReply(found: { entry: GlossaryEntry; kind: 'instrument' | 'task' }): string {
  const { entry, kind } = found;
  const lead =
    kind === 'instrument'
      ? `**${entry.name}**: a robotic instrument that ${entry.text}`
      : `**${entry.name}** (a SurgVU task label): ${entry.text}`;
  const tail =
    kind === 'instrument'
      ? 'Ask "what instruments are installed right now?" to see whether it is mounted in the current clip.'
      : `Ask "when does ${entry.name.toLowerCase()} happen?" to find it on this case's timeline.`;
  return `${lead}\n\n${tail}\n\n_General educational description, not clinical guidance._`;
}

// ---------------------------------------------------------------------------

export function converse(question: string, facts: VideoFacts): ConversationalAnswer | null {
  const raw = question || '';
  const q = normalise(raw);
  if (!q) return null;

  if (EMERGENCY.test(raw)) return { text: EMERGENCY_REPLY, kind: 'emergency', refusal: true };

  if (MEDICATION.test(raw) || PERSONAL.test(raw) || DECISION.test(raw) || OPERATIVE_INSTRUCTION.test(raw)) {
    return { text: REFUSAL, kind: 'clinical_advice', refusal: true };
  }

  if (GREETING.test(q)) return { text: greeting(facts), kind: 'greeting', refusal: false };
  if (WELLBEING.test(q)) {
    return {
      text: "I'm running fine, thanks for asking. What would you like to know about this video?",
      kind: 'wellbeing',
      refusal: false,
    };
  }
  if (THANKS.test(q)) {
    return {
      text: "You're welcome. Ask me anything else about the video, its instruments or the training tasks.",
      kind: 'thanks',
      refusal: false,
    };
  }
  if (FAREWELL.test(q)) {
    return { text: 'Goodbye, and thanks for trying it out.', kind: 'farewell', refusal: false };
  }

  if (HOWTO.test(q)) return { text: HOWTO_REPLY, kind: 'howto', refusal: false };
  if (AUTHOR.test(q)) return { text: AUTHOR_REPLY, kind: 'author', refusal: false };
  if (ABOUT.test(q)) return { text: ABOUT_REPLY, kind: 'about', refusal: false };
  if (PRIVACY.test(q)) return { text: PRIVACY_REPLY, kind: 'privacy', refusal: false };
  if (ACCURACY.test(q)) return { text: ACCURACY_REPLY, kind: 'accuracy', refusal: false };
  if (MODELS.test(q)) return { text: MODELS_REPLY, kind: 'models', refusal: false };
  if (DATASET.test(q)) return { text: DATASET_REPLY, kind: 'dataset', refusal: false };

  // "What is a needle driver?" is a definition. "Is the needle driver
  // installed now?" is about the video and belongs to the grounded path.
  if (DEFINITION.test(q) && !ABOUT_THIS_VIDEO.test(q)) {
    const found = findGlossary(q);
    if (found) return { text: definitionReply(found), kind: 'definition', refusal: false };
  }

  return null;
}

/** The reply when no path could answer. Declines rather than guesses. */
export function notTrainedFor(byokConfigured: boolean): string {
  const lines = [
    "Sorry, I'm not trained to answer that, and I don't have reliable information on it in the dataset labels or the built-in knowledge base.",
    '',
    "I won't guess. That matters most for detailed clinical questions, where a confident wrong answer could mislead. For those, please consult a qualified clinician or a peer-reviewed source.",
    '',
    'Here is what I can answer:',
    CAPABILITIES,
  ];
  if (!byokConfigured) {
    lines.push(
      '',
      'For longer explanations you can add your own Gemini API key under **Settings → Language model**. It stays in your browser.'
    );
  }
  return lines.join('\n');
}
