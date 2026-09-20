(() => {
  const shortestCountInput = document.getElementById('shortest-count');
  const durationGapInput = document.getElementById('duration-gap');
  const refreshButton = document.getElementById('refresh');
  const startRerecordButton = document.getElementById('start-rerecord');
  const statusEl = document.getElementById('qc-status');
  const errorEl = document.getElementById('qc-error');
  const lastUpdatedEl = document.getElementById('last-updated');
  const audio = document.getElementById('qc-audio');

  let keptTakes = [];
  let rerecordCandidates = [];
  let playingButton = null;

  function clampNumber(value, min, max, fallback) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  }

  async function mapWithConcurrency(items, concurrency, worker) {
    const results = new Array(items.length);
    let cursor = 0;

    async function run() {
      while (cursor < items.length) {
        const index = cursor++;
        results[index] = await worker(items[index]);
      }
    }

    const workers = Array.from(
      { length: Math.min(concurrency, items.length) },
      () => run(),
    );
    await Promise.all(workers);
    return results;
  }

  async function fetchKeptTakes() {
    const catalogResponse = await fetch('/api/catalog', { cache: 'no-store' });
    if (!catalogResponse.ok) throw new Error('Could not load the prompt catalog.');
    const catalog = await catalogResponse.json();
    const keptPrompts = (catalog.prompts || []).filter(prompt => prompt.kept_take_id);

    const results = await mapWithConcurrency(keptPrompts, 12, async (prompt) => {
      const response = await fetch(`/api/prompts/${prompt.sentence_id}`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`Could not load ${prompt.sentence_id}.`);
      const detail = await response.json();
      const validTakes = (detail.takes || [])
        .map(take => ({ ...take, duration_s: Number(take.duration_s) }))
        .filter(take => Number.isFinite(take.duration_s));
      const kept = validTakes.find(take => take.take_id === prompt.kept_take_id);
      if (!kept) return null;
      const longest = validTakes.reduce(
        (current, take) => !current || take.duration_s > current.duration_s ? take : current,
        null,
      );
      return {
        ...kept,
        text: detail.text,
        take_count: validTakes.length,
        longest_take_id: longest.take_id,
        longest_duration_s: longest.duration_s,
        duration_gap_s: Math.max(0, longest.duration_s - kept.duration_s),
      };
    });

    return results.filter(take => take && Number.isFinite(take.duration_s));
  }

  function reviewedKey(takeId) {
    return `personal-arctic-qc-reviewed:${takeId}`;
  }

  function makeRow(take, rank) {
    const row = document.createElement('div');
    row.className = 'qc-row';

    const rankEl = document.createElement('span');
    rankEl.className = 'qc-rank';
    rankEl.textContent = rank ? `#${rank}` : '—';

    const sentenceEl = document.createElement('span');
    sentenceEl.className = 'qc-sentence';
    sentenceEl.textContent = take.sentence_id;

    const promptEl = document.createElement('span');
    promptEl.className = 'qc-prompt';
    promptEl.textContent = take.text;

    const durationEl = document.createElement('span');
    durationEl.className = 'qc-duration';
    durationEl.textContent = `${take.duration_s.toFixed(2)} s`;

    const controls = document.createElement('div');
    controls.className = 'row-controls';

    const playButton = document.createElement('button');
    playButton.textContent = 'Play';
    playButton.addEventListener('click', () => playTake(take.take_id, playButton));

    const reviewLabel = document.createElement('label');
    reviewLabel.className = 'review-check';
    const reviewCheckbox = document.createElement('input');
    reviewCheckbox.type = 'checkbox';
    reviewCheckbox.dataset.takeId = take.take_id;
    reviewCheckbox.checked = localStorage.getItem(reviewedKey(take.take_id)) === '1';
    reviewCheckbox.addEventListener('change', () => {
      if (reviewCheckbox.checked) {
        localStorage.setItem(reviewedKey(take.take_id), '1');
      } else {
        localStorage.removeItem(reviewedKey(take.take_id));
      }
      renderChecks();
    });
    reviewLabel.appendChild(reviewCheckbox);
    reviewLabel.append(' Reviewed');

    controls.appendChild(playButton);
    row.appendChild(rankEl);
    row.appendChild(sentenceEl);
    row.appendChild(promptEl);
    row.appendChild(durationEl);
    row.appendChild(controls);
    row.appendChild(reviewLabel);
    return row;
  }

  function makeComparisonRow(take) {
    const row = document.createElement('div');
    row.className = 'qc-row comparison-row';

    const sentenceEl = document.createElement('span');
    sentenceEl.className = 'qc-sentence';
    sentenceEl.textContent = take.sentence_id;

    const promptEl = document.createElement('span');
    promptEl.className = 'qc-prompt';
    promptEl.textContent = take.text;

    const comparison = document.createElement('span');
    comparison.className = 'duration-comparison';
    comparison.innerHTML = `Kept <strong>${take.duration_s.toFixed(2)}s</strong> · Longest <strong>${take.longest_duration_s.toFixed(2)}s</strong> · Gap <strong>${take.duration_gap_s.toFixed(2)}s</strong>`;

    const controls = document.createElement('div');
    controls.className = 'row-controls';
    const playKept = document.createElement('button');
    playKept.textContent = 'Play kept';
    playKept.addEventListener('click', () => playTake(take.take_id, playKept));
    const playLongest = document.createElement('button');
    playLongest.textContent = 'Play longest';
    playLongest.addEventListener('click', () => playTake(take.longest_take_id, playLongest));
    controls.appendChild(playKept);
    controls.appendChild(playLongest);

    const reviewLabel = document.createElement('label');
    reviewLabel.className = 'review-check';
    const reviewCheckbox = document.createElement('input');
    reviewCheckbox.type = 'checkbox';
    reviewCheckbox.dataset.takeId = take.take_id;
    reviewCheckbox.checked = localStorage.getItem(reviewedKey(take.take_id)) === '1';
    reviewCheckbox.addEventListener('change', () => {
      if (reviewCheckbox.checked) {
        localStorage.setItem(reviewedKey(take.take_id), '1');
      } else {
        localStorage.removeItem(reviewedKey(take.take_id));
      }
      renderChecks();
    });
    reviewLabel.appendChild(reviewCheckbox);
    reviewLabel.append(' Reviewed');

    row.appendChild(sentenceEl);
    row.appendChild(promptEl);
    row.appendChild(comparison);
    row.appendChild(controls);
    row.appendChild(reviewLabel);
    return row;
  }

  function renderList(elementId, takes, ranked = false) {
    const target = document.getElementById(elementId);
    target.innerHTML = '';
    if (takes.length === 0) {
      const empty = document.createElement('p');
      empty.className = 'empty-result';
      empty.textContent = 'No recordings match this check.';
      target.appendChild(empty);
      return;
    }

    takes.forEach((take, index) => {
      target.appendChild(makeRow(take, ranked ? index + 1 : null));
    });
  }

  function renderComparisonList(takes) {
    const target = document.getElementById('gap-list');
    target.innerHTML = '';
    if (takes.length === 0) {
      const empty = document.createElement('p');
      empty.className = 'empty-result';
      empty.textContent = 'No kept takes are this much shorter than another take for the same sentence.';
      target.appendChild(empty);
      return;
    }
    takes.forEach(take => target.appendChild(makeComparisonRow(take)));
  }

  function playTake(takeId, button) {
    if (playingButton === button && !audio.paused) {
      audio.pause();
      audio.currentTime = 0;
      button.textContent = 'Play';
      playingButton = null;
      return;
    }

    if (playingButton) playingButton.textContent = 'Play';
    playingButton = button;
    button.textContent = 'Stop';
    audio.src = `/api/takes/${takeId}/audio?type=raw`;
    audio.play().catch(error => showError(`Playback failed: ${error.message}`));
  }

  audio.addEventListener('ended', () => {
    if (playingButton) playingButton.textContent = 'Play';
    playingButton = null;
  });

  function showError(message) {
    errorEl.textContent = message;
    errorEl.className = 'alert-banner warning visible';
  }

  function clearError() {
    errorEl.textContent = '';
    errorEl.className = 'alert-banner warning';
  }

  function renderChecks() {
    const count = Math.round(clampNumber(shortestCountInput.value, 1, 500, 20));
    const gapThreshold = clampNumber(durationGapInput.value, 0.1, 60, 1.0);
    shortestCountInput.value = String(count);
    durationGapInput.value = gapThreshold.toFixed(1);

    const sorted = [...keptTakes].sort((a, b) => a.duration_s - b.duration_s);
    const shortest = sorted.slice(0, count);
    const possibleTruncations = sorted
      .filter(take => take.take_count > 1 && take.duration_gap_s >= gapThreshold)
      .sort((a, b) => b.duration_gap_s - a.duration_gap_s);
    rerecordCandidates = sorted.filter(
      take => take.duration_s < 2 && localStorage.getItem(reviewedKey(take.take_id)) !== '1',
    );
    const largestGap = keptTakes.reduce(
      (largest, take) => Math.max(largest, take.duration_gap_s),
      0,
    );

    document.getElementById('kept-total').textContent = String(sorted.length);
    document.getElementById('gap-total').textContent = String(possibleTruncations.length);
    document.getElementById('shortest-duration').textContent = sorted.length ? `${sorted[0].duration_s.toFixed(2)} s` : '—';
    document.getElementById('largest-gap').textContent = sorted.length ? `${largestGap.toFixed(2)} s` : '—';
    document.getElementById('shortest-label').textContent = `Top ${Math.min(count, sorted.length)}`;
    document.getElementById('gap-label').textContent = `Gap ≥ ${gapThreshold.toFixed(1)}s`;
    document.getElementById('rerecord-label').textContent = `${rerecordCandidates.length} queued`;
    startRerecordButton.disabled = rerecordCandidates.length === 0;
    startRerecordButton.textContent = rerecordCandidates.length
      ? `Start re-recording ${rerecordCandidates.length}`
      : 'No under-2s takes to re-record';

    renderList('shortest-list', shortest, true);
    renderComparisonList(possibleTruncations);
    renderList('rerecord-list', rerecordCandidates, false);
  }

  async function refreshChecks() {
    clearError();
    refreshButton.disabled = true;
    refreshButton.textContent = 'Checking…';
    statusEl.textContent = 'Reading kept recordings from the server. Your recording session is unaffected.';

    try {
      keptTakes = await fetchKeptTakes();
      renderChecks();
      const now = new Date();
      lastUpdatedEl.textContent = `Updated ${now.toLocaleTimeString()}`;
      statusEl.textContent = `Checked ${keptTakes.length} kept recordings. Refresh again after you finish recording for the final result.`;
    } catch (error) {
      showError(error.message || String(error));
      statusEl.textContent = 'The check could not be completed.';
    } finally {
      refreshButton.disabled = false;
      refreshButton.textContent = 'Refresh checks';
    }
  }

  refreshButton.addEventListener('click', refreshChecks);
  shortestCountInput.addEventListener('change', () => {
    if (keptTakes.length) renderChecks();
  });
  durationGapInput.addEventListener('change', () => {
    if (keptTakes.length) renderChecks();
  });
  startRerecordButton.addEventListener('click', () => {
    const queue = rerecordCandidates.map(take => take.sentence_id);
    if (!queue.length) return;
    localStorage.setItem('personal-arctic-rerecord-queue', JSON.stringify(queue));
    window.location.href = '/?rerecord_queue=1&source=qc';
  });

  refreshChecks();
})();
