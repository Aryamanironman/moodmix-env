/* Shared transport and data helpers for the three direction demos. */
window.MM = (() => {
  const data = window.MOODMIX;
  const sample = data.sample;
  const tracks = data.tracks;
  const audio = new Audio(data.audioBase + sample.audio_url);
  audio.preload = "auto";
  let fake = false;
  let fakeStart = 0;
  let fakeTime = 0;
  let raf = 0;
  let frameHandler = null;

  const fmt = (seconds) => {
    if (!Number.isFinite(seconds)) return "0:00";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  };

  const stepAt = (t) => {
    const times = sample.beat_times;
    let index = 0;
    while (index + 1 < times.length && times[index + 1] <= t) index += 1;
    return sample.steps[Math.min(index, sample.steps.length - 1)];
  };

  const time = () => {
    if (fake) return fakeTime;
    if (!Number.isFinite(audio.duration) || audio.duration <= 0) return 0;
    return Math.min(1, audio.currentTime / audio.duration) * sample.duration;
  };

  const frame = () => {
    if (!frameHandler) return;
    frameHandler(stepAt(time()), time(), sample);
    if (!audio.paused || fake) raf = requestAnimationFrame(frame);
  };

  const startFake = () => {
    fake = true;
    fakeStart = performance.now() - fakeTime * 1000;
    const tick = () => {
      if (!fake) return;
      fakeTime = (performance.now() - fakeStart) / 1000;
      if (fakeTime >= sample.duration) {
        fakeTime = sample.duration;
        fake = false;
      }
      frame();
    };
    cancelAnimationFrame(raf);
    tick();
  };

  const play = () => {
    if (fake) { fake = false; }
    audio.play().then(() => {
      cancelAnimationFrame(raf);
      frame();
    }).catch(() => {
      startFake();
    });
  };

  const pause = () => {
    fake = false;
    audio.pause();
    cancelAnimationFrame(raf);
  };

  const seek = (fraction) => {
    const t = Math.max(0, Math.min(1, fraction)) * sample.duration;
    fakeTime = t;
    if (fake) {
      fakeStart = performance.now() - t * 1000;
    } else if (Number.isFinite(audio.duration) && audio.duration > 0) {
      audio.currentTime = (t / sample.duration) * audio.duration;
    }
    if (frameHandler) frameHandler(stepAt(t), t, sample);
  };

  const bind = ({ playButton, seekInput, timeLabel, onFrame }) => {
    frameHandler = onFrame;
    const toggle = () => {
      if (audio.paused && !fake) {
        playButton.textContent = "Pause";
        play();
      } else {
        playButton.textContent = "Play";
        pause();
      }
    };
    playButton.addEventListener("click", toggle);
    audio.addEventListener("play", () => { playButton.textContent = "Pause"; });
    audio.addEventListener("pause", () => { playButton.textContent = "Play"; });
    audio.addEventListener("ended", () => { playButton.textContent = "Play"; });
    seekInput.addEventListener("input", () => {
      seek(Number(seekInput.value) / 1000);
    });
    const updateClock = () => {
      if (timeLabel) timeLabel.textContent = `${fmt(time())} / ${fmt(sample.duration)}`;
      seekInput.value = String(Math.round((time() / sample.duration) * 1000));
      if (!audio.paused || fake) requestAnimationFrame(updateClock);
    };
    cancelAnimationFrame(raf);
    frameHandler(stepAt(0), 0, sample);
    updateClock();
    return {
      play, pause, seek,
      audio,
      sample,
      tracks,
      sampleAudioFailed: () => audio.error,
    };
  };

  const populateSelect = (select, selected) => {
    select.innerHTML = tracks.map(track =>
      `<option value="${track.id}" ${track.id === selected ? "selected" : ""}>${track.id}</option>`
    ).join("");
  };

  const cueOptions = (track, selected) => {
    if (!track) return "";
    return [...track.cues]
      .sort((a, b) => b.energy - a.energy)
      .map(cue => `<option value="${cue.beat}" ${cue.beat === selected ? "selected" : ""}>beat ${cue.beat} / ${cue.section} / ${cue.energy.toFixed(2)}</option>`)
      .join("");
  };

  const trackById = (id) => tracks.find(track => track.id === id);

  const wavePath = (values, x0, y0, width, height, max = 1) => {
    if (!values.length) return "";
    const mid = y0 + height / 2;
    const top = values.map((v, i) => {
      const x = x0 + (i / (values.length - 1)) * width;
      const h = (Math.max(0, Math.min(max, v)) / max) * (height / 2);
      return `${x.toFixed(1)},${(mid - h).toFixed(1)}`;
    });
    const bottom = values.map((v, i) => {
      const x = x0 + ((values.length - 1 - i) / (values.length - 1)) * width;
      const h = (Math.max(0, Math.min(max, v)) / max) * (height / 2);
      return `${x.toFixed(1)},${(mid + h).toFixed(1)}`;
    });
    return `M${top.join(" L")} L${bottom.join(" L")} Z`;
  };

  const linePath = (values, x0, y0, width, height) => {
    if (!values.length) return "";
    return values.map((v, i) => {
      const x = x0 + (i / (values.length - 1)) * width;
      const y = y0 + (1 - Math.max(0, Math.min(1, v))) * height;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  };

  return { data, sample, tracks, audio, fmt, stepAt, bind, populateSelect,
           cueOptions, trackById, wavePath, linePath };
})();
