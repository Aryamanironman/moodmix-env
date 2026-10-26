const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const els = {
  mood: $("#mood"),
  run: $("#run"),
  runSet: $("#run-set"),
  transitions: $("#transitions"),
  setStrip: $("#set-strip"),
  setChips: $("#set-chips"),
  setNote: $("#set-note"),
  policy: $$(".segmented button"),
  audio: $("#audio"),
  status: $("#status"),
  score: $("#score"),
  scoreNote: $("#score-note"),
  components: $("#components"),
  log: $("#log"),
  event: $("#event"),
  position: $("#position"),
  faderFill: $("#fader-fill"),
  faderThumb: $("#fader-thumb"),
  faderTarget: $("#fader-target"),
  faderReadout: $("#fader-readout"),
  progress: $("#progress-label"),
  curveReadout: $("#curve-readout"),
  curveTarget: $("#curve-target"),
  curveActual: $("#curve-actual"),
  curvePlayhead: $("#curve-playhead"),
  curveDot: $("#curve-dot"),
  sync: $("#sync-switch"),
  filter: $("#filter-switch"),
  beatRing: $("#beat-ring"),
  beatNumber: $("#beat-number"),
  a: {
    track: $("#a-track"), bpm: $("#a-bpm"), key: $("#a-key"),
    section: $("#a-section"), meter: $("#a-meter"), beats: $("#a-beats"),
    eq: $$("#a-eq .eq-pad"),
  },
  b: {
    track: $("#b-track"), bpm: $("#b-bpm"), key: $("#b-key"),
    section: $("#b-section"), meter: $("#b-meter"), beats: $("#b-beats"),
    eq: $$("#b-eq .eq-pad"),
  },
  pads: $$("#pads .pad"),
};

const COMPONENTS = [
  ["beat_alignment", "Beat alignment"],
  ["tempo", "Tempo"],
  ["eq_craft", "EQ craft"],
  ["bass_swap", "Bass swap"],
  ["filter_craft", "Filter"],
  ["pad_craft", "Pads"],
  ["phrase", "Phrase"],
  ["busy_hands", "Busy hands"],
  ["jitter", "Jitter"],
  ["harmonic", "Harmony"],
  ["spectral", "Spectrum"],
  ["transition_shape", "Shape"],
  ["mood", "Mood"],
  ["drum", "Drums"],
  ["smoothness", "Smoothness"],
  ["mix_quality", "Mix quality"],
  ["cut_credit", "Cut credit"],
  ["novelty", "Novelty"],
  ["end_transition", "Handover"],
  ["end_mood", "End mood"],
];

const FILTER_LABEL = ["off", "low-pass", "high-pass"];
const state = {
  data: null,
  step: -1,
  policy: "ppo",
  mode: "episode",
  loading: false,
  raf: 0,
};

function setStatus(text) {
  els.status.textContent = text;
}

function setPolicy(policy) {
  state.policy = policy;
  els.policy.forEach((button) => {
    button.classList.toggle("active", button.dataset.policy === policy);
  });
  if (state.data) {
    setStatus("Policy changed. Run again to load the new performance.");
  }
}

function eqGlyph(value) {
  return value === 0 ? "−" : value === 2 ? "+" : "·";
}

function setEq(nodes, values) {
  nodes.forEach((node, index) => {
    const value = values[index] ?? 1;
    node.textContent = eqGlyph(value);
    node.className = "eq-pad " + (value === 0 ? "cut" : value === 2 ? "boost" : "neutral");
  });
}

function setBeats(nodes, step) {
  const current = step % 4;
  nodes.forEach((node, index) => node.classList.toggle("on", index === current));
}

// Only the terms the active reward profile actually pays for get a row, so the
// panel reads the same whether the run used `expressive` or `curve`.
function shownComponents(values) {
  return COMPONENTS.filter(([name]) => name in values);
}

function makeComponents(values = {}) {
  els.components.innerHTML = "";
  shownComponents(values).forEach(([name, label]) => {
    const row = document.createElement("div");
    row.className = "component";
    row.dataset.name = name;

    const title = document.createElement("span");
    title.textContent = label;

    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("span");
    bar.appendChild(fill);

    const value = document.createElement("em");
    value.textContent = "0.00";

    row.append(title, bar, value);
    els.components.appendChild(row);
  });
}

function updateComponents(values) {
  const shown = shownComponents(values);
  const max = Math.max(
    0.05,
    ...shown.map(([name]) => Math.abs(values[name] ?? 0)),
  );
  shown.forEach(([name]) => {
    const row = els.components.querySelector(`[data-name="${name}"]`);
    if (!row) return;
    const value = values[name] ?? 0;
    row.classList.toggle("negative", value < 0);
    row.querySelector(".bar span").style.width = `${Math.min(100, Math.abs(value) / max * 100)}%`;
    row.querySelector("em").textContent = value.toFixed(2);
  });
}

function faderPoint(value) {
  return 84 - value * 72;
}

function buildCurves() {
  const steps = state.data.steps;
  const n = steps.length;
  const pathFor = (key) => steps
    .map((step, index) => {
      const x = n <= 1 ? 0 : (index / (n - 1)) * 1000;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${faderPoint(step[key]).toFixed(2)}`;
    })
    .join(" ");
  els.curveTarget.setAttribute("d", pathFor("target"));
  els.curveActual.setAttribute("d", pathFor("fader"));
}

function eventsUpTo(index) {
  const steps = state.data.steps;
  const events = [];
  let previousSync = false;
  let previousFilter = 0;
  let previousSectionA = steps[0]?.deck_a.section;
  let previousSectionB = steps[0]?.deck_b.section;

  for (let i = 0; i <= index; i += 1) {
    const step = steps[i];
    if (step.sync && !previousSync) events.push(`beat ${step.step}: deck B synced`);
    if (step.filter !== previousFilter) {
      events.push(`beat ${step.step}: filter ${FILTER_LABEL[step.filter]}`);
    }
    if (step.drum === "clap" || (step.drum === "kick" && step.step % 4 === 0)) {
      events.push(`beat ${step.step}: ${step.drum}`);
    }
    if (i > 0 && step.deck_a.section !== previousSectionA) {
      events.push(`deck A enters ${step.deck_a.section}`);
    }
    if (i > 0 && step.deck_b.section !== previousSectionB) {
      events.push(`deck B enters ${step.deck_b.section}`);
    }
    previousSync = step.sync;
    previousFilter = step.filter;
    previousSectionA = step.deck_a.section;
    previousSectionB = step.deck_b.section;
  }

  if (index >= steps.length - 1) events.push("transition complete");
  return events.slice(-5);
}

function renderStep(index) {
  const data = state.data;
  if (!data) return;
  const step = data.steps[index];
  if (!step) return;
  state.step = index;

  // Read the decks off the step, not the response: in a set the pair changes
  // at every hand-over while the player keeps running.
  els.a.track.textContent = step.deck_a.id;
  els.a.bpm.textContent = `${step.deck_a.bpm} BPM`;
  els.a.key.textContent = step.deck_a.key;
  els.a.section.textContent = step.deck_a.section;
  els.b.track.textContent = step.deck_b.id;
  els.b.bpm.textContent = `${step.deck_b.bpm} BPM`;
  els.b.key.textContent = step.deck_b.key;
  els.b.section.textContent = step.deck_b.section;

  setEq(els.a.eq, step.eq_a);
  setEq(els.b.eq, step.eq_b);
  els.a.meter.style.width = `${Math.round(step.gain_a * 100)}%`;
  els.b.meter.style.width = `${Math.round(step.gain_b * 100)}%`;
  setBeats(Array.from(els.a.beats.children), step.step);
  setBeats(Array.from(els.b.beats.children), step.step);

  const pct = Math.round(step.fader * 1000) / 10;
  els.faderFill.style.width = `${pct}%`;
  els.faderThumb.style.left = `${pct}%`;
  els.faderTarget.style.left = `${Math.round(step.target * 1000) / 10}%`;
  els.faderReadout.textContent = step.fader.toFixed(2);
  els.curveReadout.textContent = `target ${step.target.toFixed(2)} · actual ${step.fader.toFixed(2)}`;

  els.sync.classList.toggle("on", step.sync);
  els.sync.querySelector("b").textContent = step.sync ? "on" : "off";
  els.filter.classList.toggle("filter-on", step.filter !== 0);
  els.filter.querySelector("b").textContent = FILTER_LABEL[step.filter];
  els.pads.forEach((pad) => pad.classList.toggle("active", pad.dataset.drum === step.drum));

  const total = data.set ? data.set.count : 1;
  els.position.textContent = total > 1
    ? `transition ${(step.transition ?? 0) + 1}/${total} · beat ${step.step}/64`
    : `beat ${step.step} / ${data.steps.length} · bar ${step.bar}`;
  markActiveChip(step.transition ?? 0);
  els.event.textContent = eventsUpTo(index).slice(-1)[0] ?? "running";
  els.beatNumber.textContent = String((step.step % 4) + 1);
  els.beatRing.classList.remove("pulse");
  void els.beatRing.offsetWidth;
  els.beatRing.classList.add("pulse");

  const n = data.steps.length;
  const x = n <= 1 ? 0 : (index / (n - 1)) * 1000;
  els.curvePlayhead.setAttribute("x1", x);
  els.curvePlayhead.setAttribute("x2", x);
  els.curveDot.setAttribute("cx", x);
  els.curveDot.setAttribute("cy", faderPoint(step.fader));

  els.score.textContent = data.summary.total.toFixed(1);
  els.score.classList.toggle("negative", data.summary.total < 0);
  els.scoreNote.textContent = `step ${step.reward >= 0 ? "+" : ""}${step.reward.toFixed(2)} · seed ${data.seed}`;
  updateComponents(step.components);
  els.log.textContent = eventsUpTo(index).join("\n") || "Running.";
}

function stopAnimation() {
  if (state.raf) cancelAnimationFrame(state.raf);
  state.raf = 0;
}

// Steps carry their own start time in seconds, so a set with tracks at
// different tempi still maps audio time to the right beat.
function indexAtTime(seconds) {
  const steps = state.data?.steps ?? [];
  let low = 0;
  let high = steps.length - 1;
  let found = 0;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if ((steps[mid].t ?? 0) <= seconds) {
      found = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return found;
}

function tick() {
  const data = state.data;
  if (!data) return;
  const beat = data.beat_seconds || 0.5;
  const index = indexAtTime(els.audio.currentTime);
  if (index !== state.step) renderStep(index);
  const duration = Number.isFinite(els.audio.duration)
    ? els.audio.duration
    : data.steps.length * beat;
  const progress = duration > 0 ? Math.min(1, els.audio.currentTime / duration) : 0;
  els.progress.textContent = `${Math.round(progress * 100)}%`;
  if (!els.audio.paused && !els.audio.ended) {
    state.raf = requestAnimationFrame(tick);
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, (char) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char]
  ));
}

function buildSetStrip(set) {
  els.setChips.innerHTML = "";
  if (!set) {
    els.setStrip.hidden = true;
    return;
  }
  els.setStrip.hidden = false;
  set.transitions.forEach((transition) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "set-chip";
    chip.dataset.index = String(transition.index);
    chip.innerHTML = `
      <span class="chip-index">${transition.index + 1}</span>
      <span class="chip-tracks">${escapeHtml(transition.track_a)} → ${escapeHtml(transition.track_b)}</span>
      <span class="chip-score">${transition.total.toFixed(1)}</span>`;
    chip.addEventListener("click", () => {
      if (!state.data) return;
      els.audio.currentTime = transition.start_seconds;
      const index = state.data.steps.findIndex(
        (step) => (step.transition ?? 0) === transition.index,
      );
      if (index >= 0) renderStep(index);
      els.audio.play().catch(() => {});
    });
    els.setChips.appendChild(chip);
  });
  els.setNote.textContent = `${set.count} transitions · ${Math.round(set.duration_seconds / 60)} min`;
}

function markActiveChip(index) {
  Array.from(els.setChips.children).forEach((chip) => {
    chip.classList.toggle("active", Number(chip.dataset.index) === index);
  });
}

async function loadPerformance(url, label, autoplay) {
  if (state.loading) return;
  state.loading = true;
  stopAnimation();
  els.run.disabled = true;
  els.runSet.disabled = true;
  els.run.textContent = "Rendering…";
  els.runSet.textContent = "Rendering…";
  setStatus(`Rendering ${label}.`);

  try {
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `request failed (${response.status})`);

    state.data = data;
    state.step = -1;
    makeComponents({ ...data.summary.components, ...data.steps[0].components });
    buildSetStrip(data.set ?? null);
    buildCurves();
    renderStep(0);
    updateComponents(data.summary.components);
    els.score.textContent = data.summary.total.toFixed(1);
    els.scoreNote.textContent = data.set
      ? `${data.set.count} transitions · seed ${data.seed}`
      : `seed ${data.seed} · ${data.summary.completed ? "completed" : "incomplete"}`;
    els.log.textContent = data.fallback
      ? "PPO model not loaded - using the human policy.\nStart the server with --model to compare the agent."
      : `${label} ready. Press run to play.`;
    setStatus(data.fallback ? "Using human fallback policy." : `${label} ready.`);

    els.audio.pause();
    els.audio.src = data.audio_url;
    els.audio.load();
    els.audio.currentTime = 0;
    els.progress.textContent = "0%";

    if (autoplay) {
      try {
        await els.audio.play();
        setStatus(`Playing ${label}.`);
      } catch (error) {
        setStatus("Audio is ready. Press run again to play.");
        els.log.textContent = "Audio is ready but the browser blocked autoplay. Press run again.";
      }
    }
  } catch (error) {
    setStatus(`Could not render: ${error.message}`);
    els.log.textContent = `Could not render the ${label}.\n${error.message}`;
  } finally {
    state.loading = false;
    els.run.disabled = false;
    els.runSet.disabled = false;
    els.run.textContent = "Run transition";
    els.runSet.textContent = "Run set";
  }
}

async function loadEpisode(autoplay = true, mode = "episode") {
  const seed = Math.floor(Math.random() * 1_000_000);
  const query = `mood=${encodeURIComponent(els.mood.value)}&policy=${encodeURIComponent(state.policy)}&seed=${seed}`;
  if (mode === "set") {
    const count = els.transitions?.value || "4";
    state.mode = "set";
    await loadPerformance(`/api/set?${query}&transitions=${count}`, "set", autoplay);
    return;
  }
  state.mode = "episode";
  await loadPerformance(`/api/episode?${query}`, "transition", autoplay);
}

els.run.addEventListener("click", () => loadEpisode(true, "episode"));
els.runSet?.addEventListener("click", () => loadEpisode(true, "set"));
els.policy.forEach((button) => {
  button.addEventListener("click", () => setPolicy(button.dataset.policy));
});
els.mood.addEventListener("change", () => {
  if (state.data) setStatus("Mood changed. Run again to load the new transition.");
});

els.audio.addEventListener("play", () => {
  stopAnimation();
  state.raf = requestAnimationFrame(tick);
});
els.audio.addEventListener("pause", stopAnimation);
els.audio.addEventListener("ended", () => {
  stopAnimation();
  if (state.data) {
    renderStep(state.data.steps.length - 1);
    els.progress.textContent = "100%";
  }
});

function applyAvailableMoods(health) {
  const available = Array.isArray(health.available_moods) ? health.available_moods : health.moods;
  if (!Array.isArray(available) || !available.length) return;
  const all = health.moods || [];
  Array.from(els.mood.options).forEach((option) => {
    const label = option.textContent.replace(/ \u2014 no tracks$/, "");
    const playable = !all.length || available.includes(option.value);
    option.disabled = !playable;
    option.textContent = playable ? label : `${label} \u2014 no tracks`;
  });
  if (!available.includes(els.mood.value)) {
    els.mood.value = available[0];
  }
}

// `?mood=relaxed&policy=ppo&set=4` opens straight into a set, so a link can
// share one without clicking through the controls first.
function applyQuery() {
  const params = new URLSearchParams(window.location.search);
  const policy = params.get("policy");
  if (policy && ["ppo", "human", "random"].includes(policy)) setPolicy(policy);
  const mood = params.get("mood");
  if (mood && Array.from(els.mood.options).some((option) => option.value === mood && !option.disabled)) {
    els.mood.value = mood;
  }
  const set = params.get("set");
  if (!set) return "episode";
  const count = Math.max(2, Math.min(8, Number(set) || 4));
  if (els.transitions) els.transitions.value = String(count);
  return "set";
}

makeComponents();
let startMode = "episode";
fetch("/api/health")
  .then((response) => response.json())
  .then((health) => {
    applyAvailableMoods(health);
    startMode = applyQuery();
    if (!health.model_loaded) {
      els.log.textContent = "PPO model not loaded. Start the server with --model to compare the agent.";
    }
  })
  .catch(() => {})
  .finally(() => loadEpisode(false, startMode));
