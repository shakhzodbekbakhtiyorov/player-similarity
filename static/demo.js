const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

let D = null, current = -1;

// ---------- wording (mirrors labels.py) -----------------------------------
const level = z =>
  z >= 1.25 ? 'far above average' : z >= 0.45 ? 'above average' :
  z > -0.45 ? 'around average' : z > -1.25 ? 'below average' : 'far below average';

function phrase(a, b, nameA, nameB, role, agree = 0.45) {
  const sa = nameA.split(' ').pop(), sb = nameB.split(' ').pop();
  const la = level(a), lb = level(b), gap = Math.abs(a - b);
  if (la === lb) {
    const base = la === 'around average'
      ? `Both sit close to the average ${role} here.`
      : `Both are ${la} for a ${role} here.`;
    return gap > agree ? `${base} ${a > b ? sa : sb} a little more so.` : base;
  }
  if (gap <= agree) return `Very close — on the line between ${lb} and ${la} for a ${role}.`;
  const [hi, lo] = a > b ? [sa, sb] : [sb, sa];
  return `${hi} is ${gap >= 1.0 ? 'clearly' : 'somewhat'} higher than ${lo} — ` +
         `${level(Math.max(a, b))} vs ${level(Math.min(a, b))} for a ${role}.`;
}

const confidence = icc =>
  icc == null ? ['unknown', 'no cross-club estimate for this axis'] :
  icc >= 0.45 ? ['holds up', 'players tend to keep this trait after a transfer'] :
  icc >= 0.30 ? ['partly holds', 'about a third of this survives a transfer'] :
                ['season-specific', 'this mostly describes the season, not the player'];

// ---------- data ----------------------------------------------------------
async function boot() {
  const get = (f, kind) => fetch('data/' + f).then(r => r[kind]());
  const [p, s, ib, zb, ab] = await Promise.all([
    get('players.json', 'json'), get('schema.json', 'json'),
    get('index.bin', 'arrayBuffer'), get('z.bin', 'arrayBuffer'),
    get('axes.bin', 'arrayBuffer'),
  ]);
  D = {p, s, X: new Float32Array(ib), Z: new Int8Array(zb), A: new Float32Array(ab)};
  D.team = p.team_id.map(i => p.teams[i]);
  D.role = p.role_id.map(i => p.roles[i]);
  D.bucket = p.bucket_id.map(i => p.buckets[i]);
  D.key = p.player.map((n, i) => `${n} (${p.season[i]}, ${D.team[i]})`);
  D.low = D.key.map(k => k.toLowerCase());

  $('#season').insertAdjacentHTML('beforeend',
    s.seasons.map(v => `<option value="${v}">${v}</option>`).join(''));
  $('#loading').remove();
  $('#q').disabled = false;
  $('#q').focus();
}

const card = i => ({
  key: D.key[i], player: D.p.player[i], team: D.team[i], season: D.p.season[i],
  role: D.role[i], nineties: D.p.nineties[i],
});

// ---------- search --------------------------------------------------------
function search(q, limit = 12) {
  const hits = [];
  for (let i = 0; i < D.low.length; i++) if (D.low[i].includes(q)) hits.push(i);
  hits.sort((a, b) => D.p.nineties[b] - D.p.nineties[a]);
  return hits.slice(0, limit);
}

// ---------- neighbours: one mat-vec, same maths as app.py -----------------
function similar(i, {k = 10, scope = 'role', min90s = 8, season = ''} = {}) {
  const {n_rows: n, n_cols: c} = D.s, X = D.X, base = i * c;
  const me = D.p.player[i], out = [];
  for (let j = 0; j < n; j++) {
    if (j === i || D.p.player[j] === me) continue;
    if (scope === 'role' && D.role[i] !== 'OTHER'
        ? D.role[j] !== D.role[i] : D.bucket[j] !== D.bucket[i]) continue;
    if (D.p.nineties[j] < min90s) continue;
    if (season && D.p.season[j] !== season) continue;
    let dot = 0, off = j * c;
    for (let t = 0; t < c; t++) dot += X[base + t] * X[off + t];
    out.push([j, dot]);
  }
  out.sort((a, b) => b[1] - a[1]);
  return {n_candidates: out.length,
          results: out.slice(0, k).map(([j, sim], r) =>
            Object.assign(card(j), {rank: r + 1, similarity: sim}))};
}

// ---------- explanation ---------------------------------------------------
function explain(i, j) {
  const {n_cols: c, n_axes: na, z_scale: zs, style_idx: style} = D.s;
  const role = D.role[i];
  const axes = [];
  if (role === D.role[j] && D.s.axes[role] && D.A[i * na] > -900) {
    D.s.axes[role].axes.forEach((spec, n) => {
      const a = D.A[i * na + n], b = D.A[j * na + n];
      const [conf, note] = confidence(spec.icc_cross_club);
      axes.push(Object.assign({}, spec, {
        a, b, gap: Math.abs(a - b), a_level: level(a), b_level: level(b),
        sentence: phrase(a, b, D.p.player[i], D.p.player[j], role),
        confidence: conf, confidence_note: note,
      }));
    });
    axes.sort((x, y) => x.gap - y.gap);
  }

  const za = t => D.Z[i * c + t] / zs, zb = t => D.Z[j * c + t] / zs;
  const gap = t => Math.abs(za(t) - zb(t));
  const shared = t => (za(t) + zb(t)) / 2;
  const alike = style.slice().sort((x, y) =>
    (gap(x) - Math.abs(shared(x))) - (gap(y) - Math.abs(shared(y))));
  const apart = style.slice().sort((x, y) => gap(y) - gap(x));
  const feat = t => ({feature: D.s.columns[t], group: D.s.col_group[t],
                      z_a: za(t), z_b: zb(t)});

  let dot = 0, na2 = 0, nb2 = 0;
  for (let t = 0; t < c; t++) {
    dot += D.X[i * c + t] * D.X[j * c + t];
    na2 += D.X[i * c + t] ** 2; nb2 += D.X[j * c + t] ** 2;
  }
  return {a: card(i), b: card(j), role_axes: axes, axes_available: axes.length > 0,
          similarity: dot / (Math.sqrt(na2 * nb2) + 1e-12),
          most_alike: alike.slice(0, 6).map(feat),
          most_different: apart.slice(0, 6).map(feat)};
}

// ---------- ui ------------------------------------------------------------
let timer;
$('#q').addEventListener('input', e => {
  clearTimeout(timer);
  const q = e.target.value.trim().toLowerCase();
  if (q.length < 2) { $('#hits').innerHTML = ''; return; }
  timer = setTimeout(() => {
    $('#hits').innerHTML = search(q).map(i => `
      <li data-i="${i}"><span>${esc(D.p.player[i])}</span>
        <span class="tag">${esc(D.team[i])} &middot; ${esc(D.p.season[i])} &middot;
          ${esc(D.role[i])} &middot; ${D.p.nineties[i]} 90s</span></li>`).join('');
  }, 120);
});

$('#hits').addEventListener('click', e => {
  const li = e.target.closest('li'); if (!li) return;
  $('#hits').innerHTML = ''; $('#q').value = D.key[+li.dataset.i];
  select(+li.dataset.i);
});

['#scope', '#k', '#min90s', '#season'].forEach(s =>
  $(s).addEventListener('change', () => current >= 0 && select(current)));

function select(i) {
  current = i;
  $('#explainPanel').hidden = true;
  const d = similar(i, {k: +$('#k').value, scope: $('#scope').value,
                        min90s: +$('#min90s').value, season: $('#season').value});
  const q = card(i);
  $('#resultsTitle').textContent =
    `Most similar to ${q.player} — ${q.team}, ${q.season} (${q.role})`;
  $('#rows').innerHTML = d.results.map(r => `
    <tr data-i="${D.key.indexOf(r.key)}">
      <td class="num">${r.rank}</td><td>${esc(r.player)}</td>
      <td>${esc(r.team)}</td><td class="num">${esc(r.season)}</td>
      <td class="simcell">
        <div class="simbar" style="width:${Math.max(0, r.similarity) * 100}%"></div>
        <span class="tag num">${r.similarity.toFixed(3)}</span></td>
    </tr>`).join('') || `<tr><td colspan="5" class="empty">No candidates</td></tr>`;
  $('#candNote').textContent =
    `${d.n_candidates.toLocaleString()} candidate player-seasons. Click a row to see why.`;
  $('#resultsPanel').hidden = false;
}

$('#rows').addEventListener('click', e => {
  const tr = e.target.closest('tr'); if (!tr || !tr.dataset.i) return;
  [...$('#rows').children].forEach(r => r.classList.toggle('sel', r === tr));
  render(explain(current, +tr.dataset.i));
});

const pos = v => (Math.max(-3, Math.min(3, v)) + 3) / 6 * 100;

function render(d) {
  $('#explainTitle').textContent = `${d.a.player} vs ${d.b.player}`;
  if (d.axes_available) {
    $('#axes').innerHTML = d.role_axes.map((a, n) => {
      const pa = pos(a.a), pb = pos(a.b), lo = Math.min(pa, pb);
      const cls = a.confidence === 'holds up' ? 'hi'
                : a.confidence === 'season-specific' ? 'lo' : '';
      return `<div class="axis">
        <div class="axis-head">
          <span class="axis-name">${esc(a.title)}</span>
          <span class="conf ${cls}" title="${esc(a.confidence_note)}">${esc(a.confidence)}</span>
        </div>
        <p class="sentence">${esc(a.sentence)}</p>
        <div class="track">
          <div class="seg" style="left:${lo}%;width:${Math.abs(pa - pb)}%"></div>
          <div class="dot a" style="left:${pa}%"></div>
          <div class="dot b" style="left:${pb}%"></div>
        </div>
        ${n === 0 ? `<div class="scale"><span>less</span>
          <span>average ${esc(d.a.role)}</span><span>more</span></div>` : ''}
        <div class="vals">
          <b>${esc(d.a.player)}</b> ${a.a_level} (${a.a.toFixed(2)}) &nbsp;·&nbsp;
          <b>${esc(d.b.player)}</b> ${a.b_level} (${a.b.toFixed(2)})<br>
          Higher means more: ${esc((a.high || []).join(', ') || '—')}${
            a.low && a.low.length ? `. Lower means more: ${esc(a.low.join(', '))}.` : '.'}
        </div></div>`;
    }).join('');
    $('#legend').innerHTML =
      `<span><i class="key" style="background:var(--accent)"></i>${esc(d.a.player)}</span>
       <span><i class="key" style="background:transparent;border:2px solid var(--b)"></i>${esc(d.b.player)}</span>
       <span>Sorted most-alike first. Scale is standard deviations from the average ${esc(d.a.role)}.</span>`;
    $('#axisNote').innerHTML =
      `"Holds up" vs "season-specific" is how much of a player's position on that
       row survives a move to another club. Treat season-specific rows as a
       description of that season, not a prediction about the player.`;
  } else {
    $('#axes').innerHTML = `<p class="empty">These two play different fine roles
      (${esc(d.a.role)} vs ${esc(d.b.role)}), so there is no shared axis model.
      The feature breakdown below still applies.</p>`;
    $('#legend').innerHTML = ''; $('#axisNote').textContent = '';
  }
  const li = f => `<li><span>${esc(f.feature)} <span class="tag">${esc(f.group)}</span></span>
    <span class="tag num">${f.z_a.toFixed(2)} / ${f.z_b.toFixed(2)}</span></li>`;
  $('#alike').innerHTML = d.most_alike.map(li).join('');
  $('#diff').innerHTML = d.most_different.map(li).join('');
  $('#explainPanel').hidden = false;
  $('#explainPanel').scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

$('#q').disabled = true;
boot();
