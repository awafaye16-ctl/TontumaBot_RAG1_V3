/* ═══════════════════════════════════════════════════════════════════════════
   TontumaBot — logique de la borne (indépendante de l'habillage)

   Pilote l'écran (conversation, barre d'état, clavier tactile), le micro, le
   dialogue SSE avec le backend et la machine à états. Le châssis — boîtier 3D
   (borne.html) ou panneau CSS (borne-simple.html) — s'abonne via `onRender`
   et déclenche les actions via `Borne.toggleVoice()` / `Borne.pressStartStop()`.

   API :
     Borne.init({ onRender, onKey, onHealth })
     Borne.pressLang('wo' | 'fr')        boutons 1 et 2 — parler en wolof / en français
     Borne.pressStartStop()              bouton 3 — marche/arrêt et stop (appui court)
     Borne.holdStart() / holdEnd(fire)   bouton 3 — gestion de l'appui long
     Borne.powerOn() / powerOff()
     Borne.info()                        instantané de l'état
   ═══════════════════════════════════════════════════════════════════════════ */

window.Borne = (function () {
  'use strict';

  const $ = id => document.getElementById(id);

  // ── Machine à états ──
  //   off        borne éteinte (veille)
  //   idle       allumée, en attente d'une question
  //   listening  micro ouvert (mode vocal)
  //   processing requête en cours (pipeline RAG en SSE)
  //   speaking   lecture de la réponse audio
  const S = { OFF: 'off', IDLE: 'idle', LISTENING: 'listening', PROCESSING: 'processing', SPEAKING: 'speaking' };

  // Limites de session (usage public : on libère la borne)
  const MAX_MESSAGES  = 10;      // 10 questions par session, puis extinction
  const IDLE_TIMEOUT  = 600000;  // 10 min sans interaction → extinction
  const CLOSING_DELAY = 6000;    // délai de lecture avant l'extinction finale

  const LANGS = {
    wo: { code: 'wo', nom: 'wolof',    drapeau: '🇸🇳', ecoute: 'Waxal ci wolof…' },
    fr: { code: 'fr', nom: 'français', drapeau: '🇫🇷', ecoute: 'Parlez en français…' },
  };

  let sessionId  = null;    // identifie la mémoire conversationnelle côté serveur
  let memCount   = 0;       // messages actuellement en mémoire (affichage)
  let memMax     = 10;      // renseigné par la réponse du serveur

  let state      = S.OFF;
  let micLang    = null;    // langue de l'enregistrement en cours
  let lastLang   = 'wo';    // dernière langue utilisée (affichage)
  let lastSource = 'text';  // 'voice' | 'text' — décide de la lecture vocale
  let msgCount  = 0;       // questions posées dans la session courante
  let abortCtl  = null;    // annulation de la requête en cours
  let player    = null;    // <audio> réutilisé pour toutes les réponses TTS
  let speaking  = false;   // une lecture est-elle en cours sur cet élément ?
  let unlocked  = false;   // l'élément a-t-il déjà démarré sur un geste ?
  let idleTimer = null;    // retour en veille après inactivité

  // Diffusion de la synthèse : le serveur envoie la réponse par morceaux, et
  // la borne parle dès le premier au lieu d'attendre l'énoncé complet.
  let fileMorceaux  = [];     // morceaux reçus, pas encore lus
  let morceauxLus   = 0;      // morceaux déjà lus jusqu'au bout
  let morceauxTotal = 0;      // annoncé par le serveur (0 = pas de diffusion)
  let lectureAbandonnee = false;  // STOP, extinction, ou échec de lecture
  let lectureEchouee    = false;  // le navigateur a refusé : bouton ▶ à l'arrivée
  let diffusionRecue    = false;  // au moins un morceau est arrivé par le flux
  let bulleReponse  = null;   // bulle de réponse, créée dès le texte connu

  let hooks = { onRender: null, onKey: null, onHealth: null };

  let screen, standby, session, convo, statusEl, recTime, counter, keyboard, kbInput,
      vizEl, toastEl, memEl;

  // ═══════════════════════════════════════════════════════════════════
  //  Utilitaires
  // ═══════════════════════════════════════════════════════════════════
  const esc = s => (s == null ? '' : String(s))
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  function toast(msg) {
    if (!toastEl) return;
    toastEl.textContent = msg;
    toastEl.classList.add('show');
    clearTimeout(toastEl._id);
    toastEl._id = setTimeout(() => toastEl.classList.remove('show'), 3600);
  }

  // Retour sonore d'appui, comme sur une vraie borne.
  function beep(freq = 880, ms = 90) {
    try {
      const ctx = beep._ctx || (beep._ctx = new (window.AudioContext || window.webkitAudioContext)());
      if (ctx.state === 'suspended') ctx.resume();
      const osc = ctx.createOscillator(), gain = ctx.createGain();
      osc.type = 'square';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.05, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + ms / 1000);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + ms / 1000);
    } catch (_) { }
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Lecture audio des réponses
  //
  //  Safari — et Firefox en politique stricte — n'autorisent une lecture
  //  différée que sur un élément <audio> déjà démarré pendant un geste
  //  utilisateur. Comme la réponse arrive des dizaines de secondes après
  //  l'appui sur START, on amorce l'élément avec un silence dès le premier
  //  appui sur un bouton, puis on réutilise CE MÊME élément pour toutes les
  //  réponses. Sinon la lecture est refusée et l'usager n'entend rien.
  // ═══════════════════════════════════════════════════════════════════
  const SILENCE_WAV = 'data:audio/wav;base64,UklGRnQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YVAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==';

  function getPlayer() {
    if (!player) {
      player = new Audio();
      player.preload = 'auto';
    }
    return player;
  }

  function unlockAudio() {
    if (unlocked) return;
    const p = getPlayer();
    try {
      p.src = SILENCE_WAV;
      const r = p.play();
      if (r && r.then) {
        r.then(() => { p.pause(); p.currentTime = 0; unlocked = true; })
         .catch(() => { /* refusé : on retentera au prochain appui */ });
      } else {
        p.pause();
        unlocked = true;
      }
    } catch (_) { }
  }

  function addMsg(role, html) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    div.innerHTML = html;
    convo.appendChild(div);
    scrollBas();
    return div;
  }

  function setStatus(txt) { statusEl.innerHTML = txt; }

  // ── Session serveur : porte la mémoire conversationnelle ──
  function nouvelleSession() {
    try { return crypto.randomUUID(); }
    catch (_) { return 'borne-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8); }
  }

  function oublierSession() {
    if (!sessionId) return;
    // `keepalive` : la requête survit à l'extinction de la borne.
    try { fetch('/session/' + sessionId, { method: 'DELETE', keepalive: true }); } catch (_) { }
    sessionId = null;
    memCount  = 0;
  }

  // Défilement immédiat : dans la borne 3D la dalle est un CSS3DObject dont la
  // transformation est réécrite à chaque image, ce qui avale un défilement animé.
  function scrollBas() {
    const bas = () => {
      try { convo.scrollTo({ top: convo.scrollHeight, behavior: 'instant' }); }
      catch (_) { convo.scrollTop = convo.scrollHeight; }
    };
    bas();
    requestAnimationFrame(bas);   // après la mise en page définitive
    setTimeout(bas, 150);         // après l'arrivée d'une image (QR code)
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Inactivité et quota de session
  // ═══════════════════════════════════════════════════════════════════
  function resetIdleTimer() {
    clearTimeout(idleTimer);
    if (state === S.OFF) return;
    idleTimer = setTimeout(() => {
      // Occupée (écoute, traitement, lecture) : on ne coupe pas, on repousse.
      if (state !== S.IDLE) { resetIdleTimer(); return; }
      toast('⏱️ 10 min sans activité — la borne s’éteint');
      addMsg('bot', '<span class="tag">Session terminée</span>' +
        '⏱️ Aucune activité depuis 10 minutes — la borne se met en veille. Jërëjëf !');
      setStatus('⏱️ Extinction pour inactivité…');
      setTimeout(powerOff, 2500);
    }, IDLE_TIMEOUT);
  }

  function renderCounter() {
    if (memEl) {
      memEl.classList.toggle('show', state !== S.OFF);
      memEl.textContent = '🧠 ' + memCount + ' / ' + memMax;
      memEl.title = 'Mémoire conversationnelle : ' + memCount + ' message(s) sur ' +
                    memMax + ' — remise à zéro une fois la limite atteinte';
    }
    const left = MAX_MESSAGES - msgCount;
    counter.classList.toggle('show', state !== S.OFF);
    counter.classList.toggle('warn', left <= 3 && left > 0);
    counter.classList.toggle('full', left <= 0);
    counter.textContent = '💬 ' + msgCount + ' / ' + MAX_MESSAGES;
    counter.title = left > 0 ? left + ' question(s) restante(s)' : 'Session complète';
  }

  // Quota atteint : on laisse lire (et écouter) la dernière réponse, puis on
  // éteint pour libérer la borne.
  function maybeEndSession() {
    if (msgCount < MAX_MESSAGES) return false;
    addMsg('bot', '<span class="tag">Session terminée</span>' +
      '✅ Vous avez posé les <b>' + MAX_MESSAGES + ' questions</b> de cette session. ' +
      'La borne va s’éteindre — appuyez sur <b>MARCHE</b> pour en commencer une nouvelle. Jërëjëf !');
    setStatus('🔚 Session terminée — extinction…');
    clearTimeout(idleTimer);
    setTimeout(powerOff, CLOSING_DELAY);
    return true;
  }

  function remainingStatus(prefix) {
    const left = MAX_MESSAGES - msgCount;
    setStatus(prefix + ' — <b>' + left + '</b> restante' + (left > 1 ? 's' : ''));
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Rendu
  // ═══════════════════════════════════════════════════════════════════
  function info() {
    return {
      state, micLang, lastLang, msgCount,
      on:    state !== S.OFF,
      rec:   state === S.LISTENING,
      busy:  state === S.PROCESSING || state === S.SPEAKING,
      max:   MAX_MESSAGES,
      left:  MAX_MESSAGES - msgCount,
      level: micLevel,
    };
  }

  function render() {
    const i = info();

    screen.classList.toggle('off', !i.on);
    standby.classList.toggle('hide', i.on);
    session.classList.toggle('show', i.on);

    // Le clavier reste accessible en permanence : la voix et la saisie sont
    // deux entrées parallèles, plus un mode à activer.
    keyboard.classList.toggle('show', i.on);
    recTime.classList.toggle('show', i.rec);
    renderCounter();
    if (!i.rec) stopViz();

    if (hooks.onRender) hooks.onRender(i);
  }

  function setState(s) { state = s; render(); resetIdleTimer(); }

  // ═══════════════════════════════════════════════════════════════════
  //  Allumage / extinction
  // ═══════════════════════════════════════════════════════════════════
  function powerOn() {
    msgCount  = 0;                // nouvelle session
    memCount  = 0;
    sessionId = nouvelleSession();
    setState(S.IDLE);
    convo.innerHTML = '';
    addMsg('bot',
      '<span class="tag">Borne active</span>' +
      'Salam aléikum ! 👋 Je vous accompagne dans vos démarches administratives.<br>' +
      'Appuyez sur <b>🇸🇳 WOLOF</b> ou <b>🇫🇷 FRANÇAIS</b> pour parler — ou saisissez ' +
      'votre question au clavier.' +
      '<br><small style="color:var(--muted)">Session de <b>' + MAX_MESSAGES +
      ' questions</b> · extinction automatique après 10 min d’inactivité</small>');
    setStatus('🎙️ <b>WOLOF</b> ou <b>FRANÇAIS</b> pour parler · ⌨️ ou tapez votre question');
    beep(660, 70);
    setTimeout(() => beep(990, 90), 90);
    kbInput.focus();
  }

  function powerOff() {
    cancelCurrent();
    stopRecording(true);
    stopSpeaking();
    oublierSession();             // la mémoire ne survit pas à l'usager
    kbInput.value = '';
    setState(S.OFF);
    clearTimeout(idleTimer);
    beep(520, 120);
  }

  function cancelCurrent() { if (abortCtl) { abortCtl.abort(); abortCtl = null; } }
  function stopSpeaking() {
    if (player && speaking) { try { player.pause(); } catch (_) { } }
    speaking = false;
    // Les morceaux encore en vol ne doivent pas relancer la lecture : un STOP
    // arrête la réponse, pas seulement la phrase en cours.
    lectureAbandonnee = true;
    fileMorceaux = [];
    progresCacher();
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Boutons 1 et 2 — PARLER EN WOLOF / EN FRANÇAIS
  //
  //  Chaque bouton est un « appuyer pour parler » : il choisit la langue ET
  //  démarre l'écoute. Un second appui sur le même bouton envoie ; un appui
  //  sur l'autre langue relance l'écoute dans celle-ci (erreur de bouton).
  //  La langue sert aussi côté serveur à choisir le moteur STT, et évite une
  //  détection automatique sur la transcription.
  // ═══════════════════════════════════════════════════════════════════
  function pressLang(lang) {
    unlockAudio();
    resetIdleTimer();

    const L = LANGS[lang] || LANGS.wo;
    lastLang = L.code;

    if (state === S.PROCESSING || state === S.SPEAKING) {
      toast('⏳ Réponse en cours — appuyez sur <b>STOP</b> pour l’interrompre');
      return;
    }

    beep(L.code === 'wo' ? 880 : 1180, 70);

    if (state === S.LISTENING) {
      if (micLang === L.code) { stopRecording(); return; }   // même bouton → envoyer
      stopRecording(true);                                    // autre langue → reprendre
      setTimeout(() => startRecording(L.code), 120);
      toast(L.drapeau + ' Reprise en ' + L.nom);
      return;
    }

    if (msgCount >= MAX_MESSAGES) {
      toast('🔚 Session complète (' + MAX_MESSAGES + ' questions) — la borne s’éteint');
      return;
    }

    // Borne en veille : un appui sur une langue vaut mise en marche + écoute.
    if (state === S.OFF) {
      powerOn();
      setTimeout(() => startRecording(L.code), 250);
      return;
    }

    startRecording(L.code);
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Bouton 3 — MARCHE / ARRÊT (et STOP)
  //    éteinte              → allumer
  //    enregistrement       → arrêter et envoyer
  //    traitement / lecture → interrompre
  //    prête + texte saisi  → envoyer la saisie
  //    appui long (1,5 s)   → éteindre la borne
  // ═══════════════════════════════════════════════════════════════════
  function pressStartStop() {
    unlockAudio();
    resetIdleTimer();

    if (state === S.OFF) { powerOn(); return; }

    beep(760, 70);

    if (state === S.LISTENING) { stopRecording(); return; }

    if (state === S.PROCESSING) {
      cancelCurrent();
      removeThinking();
      addMsg('bot err', '⏹ Demande interrompue.');
      if (msgCount > 0) msgCount--;        // interruption : quota non consommé
      setState(S.IDLE);
      setStatus('Interrompu — prêt pour une nouvelle question');
      return;
    }

    if (state === S.SPEAKING) {
      stopSpeaking();
      setState(S.IDLE);
      if (!maybeEndSession()) setStatus('⏹ Lecture arrêtée — prêt');
      return;
    }

    // IDLE
    if (msgCount >= MAX_MESSAGES) {
      toast('🔚 Session complète (' + MAX_MESSAGES + ' questions) — la borne s’éteint');
      return;
    }
    const txt = kbInput.value.trim();
    if (!txt) {
      toast('🎙️ Appuyez sur <b>WOLOF</b> ou <b>FRANÇAIS</b> pour parler, ou tapez votre question');
      kbInput.focus();
      return;
    }
    kbInput.value = '';
    sendText(txt);
  }

  // ── Appui long sur START/STOP = extinction ──
  let holdTimer = null, heldOff = false;

  function holdStart() {
    unlockAudio();
    heldOff = false;
    if (state === S.OFF) return;
    holdTimer = setTimeout(() => {
      heldOff = true;
      toast('⏻ Borne éteinte');
      powerOff();
    }, 1500);
  }

  function holdEnd(fire) {
    clearTimeout(holdTimer);
    if (fire && !heldOff) pressStartStop();
    heldOff = false;
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Clavier tactile (mode texte)
  // ═══════════════════════════════════════════════════════════════════
  function buildKeyboard() {
    document.querySelectorAll('.kb-row[data-keys]').forEach(row => {
      row.dataset.keys.split(' ').forEach(k => {
        const b = document.createElement('button');
        b.className = 'key';
        b.textContent = k;
        b.addEventListener('click', () => insert(k));
        row.appendChild(b);
      });
    });

    const last = document.createElement('div');
    last.className = 'kb-row';
    [
      ['?',         'key',       () => insert('?')],
      ['␣ espace',  'key space', () => insert(' ')],
      ['⌫',         'key wide',  () => { kbInput.value = kbInput.value.slice(0, -1); }],
      ['Envoyer',   'key go',    () => pressStartStop()],
    ].forEach(([label, cls, fn]) => {
      const b = document.createElement('button');
      b.className = cls;
      b.textContent = label;
      b.addEventListener('click', () => { beep(1200, 40); fn(); resetIdleTimer(); });
      last.appendChild(b);
    });
    keyboard.appendChild(last);

    // Bip sur les touches alphabétiques (déléguée).
    keyboard.addEventListener('click', e => {
      if (e.target.classList.contains('key') && !e.target.classList.contains('go')) beep(1400, 30);
      resetIdleTimer();
    });
  }

  function insert(ch) { kbInput.value += ch; }

  // ═══════════════════════════════════════════════════════════════════
  //  Micro — enregistrement (MediaRecorder) + niveau sonore
  // ═══════════════════════════════════════════════════════════════════
  let mediaRecorder = null, chunks = [], stream = null,
      recStart = 0, recTimerId = null,
      audioCtx = null, analyser = null, vizId = null, micLevel = 0;

  const VIZ_BARS = 22;

  function extPourType(type) {
    if (!type) return 'webm';
    if (type.includes('mp4') || type.includes('aac')) return 'm4a';
    if (type.includes('ogg')) return 'ogg';
    return 'webm';
  }

  // `navigator.mediaDevices` n'existe qu'en contexte sécurisé (HTTPS ou
  // localhost) : distinguer ce cas d'un défaut de support évite un message
  // trompeur sur la borne.
  function raisonMicroIndisponible() {
    if (!window.isSecureContext) {
      return "le micro exige une origine sécurisée (HTTPS ; en HTTP seul localhost est autorisé)";
    }
    return "ce navigateur ne gère pas l'enregistrement direct";
  }

  function fmtTime(ms) {
    const s = Math.floor(ms / 1000);
    return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
  }

  function startViz(srcStream) {
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const src = audioCtx.createMediaStreamSource(srcStream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      src.connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);
      const bars = vizEl.children;
      const tick = () => {
        analyser.getByteFrequencyData(data);
        let sum = 0;
        for (let i = 0; i < bars.length; i++) {
          const v = data[i % data.length] / 255;
          sum += v;
          bars[i].style.height = Math.max(4, v * 28) + 'px';
        }
        micLevel = sum / bars.length;   // exploité par l'habillage 3D
        vizId = requestAnimationFrame(tick);
      };
      tick();
    } catch (_) { }
  }

  function stopViz() {
    cancelAnimationFrame(vizId);
    vizId = null;
    micLevel = 0;
    if (vizEl) Array.from(vizEl.children).forEach(b => b.style.height = '4px');
    if (audioCtx) { try { audioCtx.close(); } catch (_) { } audioCtx = null; }
  }

  async function startRecording(lang) {
    const L = LANGS[lang] || LANGS[lastLang] || LANGS.wo;
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      toast('🎙️ Micro indisponible : ' + raisonMicroIndisponible() + '.');
      setStatus('❌ Micro indisponible — utilisez le clavier tactile');
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      toast('❌ Micro inaccessible (' + (e.name || 'erreur') + ')');
      setStatus('❌ Micro refusé — autorisez l’accès ou passez au clavier');
      return;
    }

    micLang  = L.code;
    lastLang = L.code;

    chunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    mediaRecorder.onstop = () => {
      const type = mediaRecorder.mimeType || 'audio/webm';
      const blob = new Blob(chunks, { type });
      stream.getTracks().forEach(t => t.stop());
      stream = null;
      clearInterval(recTimerId);
      stopViz();
      const lang = micLang || 'wo';
      micLang = null;
      if (mediaRecorder._cancelled || blob.size < 1200) {
        setState(S.IDLE);
        setStatus(mediaRecorder._cancelled ? 'Enregistrement annulé' : '🎙️ Rien d’audible — réessayez');
        return;
      }
      sendAudio(new File([blob], 'borne.' + extPourType(type), { type }), lang);
    };

    mediaRecorder.start();
    recStart = Date.now();
    recTime.textContent = '0:00';
    recTimerId = setInterval(() => recTime.textContent = fmtTime(Date.now() - recStart), 250);
    startViz(stream);
    setState(S.LISTENING);
    setStatus('🔴 ' + L.drapeau + ' <b>' + L.ecoute + '</b> — appuyez à nouveau sur <b>' +
              L.nom.toUpperCase() + '</b> ou sur <b>STOP</b> pour envoyer');
  }

  function stopRecording(cancel) {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder._cancelled = !!cancel;
      mediaRecorder.stop();
    } else if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Client SSE — protocole commun à l'interface classique
  // ═══════════════════════════════════════════════════════════════════
  const STEP_LABELS = {
    start:         '⏳ Démarrage…',
    stt:           '🎙️ Transcription de votre voix…',
    stt_wo:        '🎙️🇸🇳 Transcription du wolof…',
    stt_fr:        '🎙️🇫🇷 Transcription du français…',
    detect:        '🌐 Détection de la langue…',
    translate_in:  '🔄 Traduction wolof → français…',
    intent:        '🧭 Analyse de la demande…',
    retrieval:     '🔍 Recherche dans les documents…',
    llm:           '🤖 Rédaction de la réponse…',
    translate_out: '🔄 Traduction français → wolof…',
    tts:           '🔊 Synthèse vocale…',
    tts_chunk:     '🔊 Synthèse vocale…',
  };

  // ═══════════════════════════════════════════════════════════════════
  //  Progression du pipeline
  //
  //  Chaque étape porte un jalon, place estimée de sa fin sur la jauge. Les
  //  poids ne sont pas uniformes : sur une question vocale, la synthèse pèse
  //  à elle seule plus que tout le reste (48 s contre 11 s pour la chaîne
  //  RAG), et un découpage en parts égales donnerait une jauge qui bondit
  //  jusqu'à 90 % puis semble bloquée.
  //
  //  L'aiguille ne saute pas d'un jalon à l'autre : elle s'en approche par
  //  fractions, vite d'abord puis de plus en plus lentement. Une étape de
  //  vingt secondes garde ainsi une jauge qui avance — sans jamais atteindre
  //  le jalon suivant avant que le serveur ne l'annonce, donc sans mentir.
  // ═══════════════════════════════════════════════════════════════════
  const JALONS = {
    start: 3, stt: 10, stt_wo: 10, stt_fr: 10, detect: 14, translate_in: 20,
    intent: 24, retrieval: 32, llm: 44, translate_out: 56, tts: 62,
  };
  const JALON_TTS = 62;        // début de la synthèse
  const JALON_LECTURE = 96;    // dernier morceau synthétisé

  let progresEl = null, progresBarEl = null;
  let progresCible = 0, progresAffiche = 0, progresTimer = null;
  let progresFacteur = 1, avecTTS = false;

  function progresInit() {
    const barre = document.querySelector('.statusbar');
    if (!barre || progresEl) return;
    progresEl = document.createElement('div');
    progresEl.className = 'progress';
    progresBarEl = document.createElement('i');
    progresEl.appendChild(progresBarEl);
    barre.appendChild(progresEl);
  }

  function progresAnimer() {
    if (progresTimer) return;
    progresTimer = setInterval(() => {
      const ecart = progresCible - progresAffiche;
      if (Math.abs(ecart) < 0.05) return;
      progresAffiche += ecart * 0.07;
      progresBarEl.style.width = progresAffiche.toFixed(2) + '%';
    }, 80);
  }

  function progresRAZ() {
    if (!progresEl) return;
    progresCible = progresAffiche = 0;
    progresFacteur = 1;
    avecTTS = false;
    progresBarEl.style.width = '0%';
    progresEl.classList.add('show');
    progresEl.classList.remove('lecture');
    progresAnimer();
  }

  function progresVers(pct) {
    if (!progresEl) return;
    progresCible = Math.max(progresCible, pct);   // une jauge ne recule pas
  }

  function progresLecture(lus, total) {
    if (!progresEl) return;
    progresEl.classList.add('show', 'lecture');
    // La lecture peut suivre un STOP (bouton ▶) : l'animation a pu être
    // arrêtée entre-temps.
    progresAnimer();
    progresVers(total ? JALON_LECTURE + (100 - JALON_LECTURE) * (lus / total) : 100);
  }

  function progresCacher() {
    if (!progresEl) return;
    clearInterval(progresTimer);
    progresTimer = null;
    progresEl.classList.remove('show');
  }

  // La course s'arrête à un jalon différent selon le travail demandé : la
  // synthèse (qui pousse ensuite la jauge morceau par morceau), la traduction
  // de sortie pour une question wolof sans TTS, ou la rédaction pour une
  // question française. Les jalons sont mis à l'échelle de cette fin-là, sinon
  // la réponse arrive sur une barre aux deux tiers.
  function progresEchelle(synthese, langue) {
    if (synthese) return 1;                       // 62 → 96 tenu par les morceaux
    const dernier = langue === 'fr' ? JALONS.llm : JALONS.translate_out;
    return 95 / dernier;
  }

  function progresEtape(step, data) {
    if (step === 'start')  { avecTTS = !!(data && data.tts); progresFacteur = progresEchelle(avecTTS, null); }
    if (step === 'detect') progresFacteur = Math.max(progresFacteur, progresEchelle(avecTTS, data && data.lang));
    if (step === 'tts_chunk') {
      // La synthèse occupe la fin de la jauge, morceau par morceau.
      const part = data && data.total ? data.index / data.total : 0;
      progresVers(JALON_TTS + (JALON_LECTURE - JALON_TTS) * part);
      return;
    }
    if (JALONS[step] !== undefined) progresVers(JALONS[step] * progresFacteur);
  }

  function addThinking() {
    removeThinking();
    bulleReponse = null;
    const d = addMsg('bot', '<span class="tag">Traitement</span><span id="think-txt">⏳ Démarrage…</span>');
    d.id = 'thinking';
    progresRAZ();
  }
  function removeThinking() { const t = $('thinking'); if (t) t.remove(); }

  function updateThinking(step, data) {
    if (step === 'stt' && data && data.lang) step = 'stt_' + data.lang;
    progresEtape(step, data);

    let label = STEP_LABELS[step] || esc(step);
    if (step === 'tts_chunk' && data && data.total > 1) {
      label = '🔊 Synthèse vocale… (' + data.index + '/' + data.total + ')';
    }
    const t = $('think-txt');
    if (t) t.innerHTML = label;
    setStatus(label);
  }

  async function consumeSSE(res, { onStatus, onResult, onError }) {
    const ct = res.headers.get('content-type') || '';
    if (!res.body || !ct.includes('text/event-stream')) {
      let msg = 'Erreur ' + res.status;
      try { const j = await res.json(); msg = j.detail || msg; } catch (_) { }
      onError(msg);
      return;
    }
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop();              // fragment incomplet conservé
      for (const block of blocks) {
        if (!block.trim()) continue;
        const ev = (block.match(/^event: (.*)$/m) || [])[1] || 'message';
        let data = {};
        try { data = JSON.parse((block.match(/^data: (.*)$/m) || [])[1] || '{}'); } catch (_) { }
        if (ev === 'status')      onStatus(data);
        else if (ev === 'result') onResult(data);
        else if (ev === 'error')  onError(data.message || 'Erreur');
      }
    }
  }

  // Affichage seul : appelé une première fois quand le serveur annonce la
  // synthèse — le texte est alors connu, l'audio non — puis une seconde fois
  // à l'arrivée du résultat complet, qui apporte le QR code. La bulle est
  // enrichie sur place plutôt que dupliquée.
  function afficherReponse(data) {
    const t = data.trace || {};
    const langue = t.input_lang || data.lang;
    const isWo = langue === 'wo';
    let html = '<span class="tag">' + (isWo ? 'Tontu ci wolof 🇸🇳' : 'Réponse') + '</span>';
    if (data.response_fr && data.response_wo) html += '<div class="fr">FR : ' + esc(data.response_fr) + '</div>';
    html += '<div>' + esc(data.response || data.response_fr || '') + '</div>';
    if (data.qr_code) {
      html += '<img class="qr" src="data:image/png;base64,' + esc(data.qr_code) + '" alt="QR code de la procédure">' +
              '<div style="text-align:center;font-size:12px;color:var(--muted);margin-top:6px;">' +
              '📱 Scannez pour emporter la procédure</div>';
    }
    if (bulleReponse) bulleReponse.innerHTML = html;
    else bulleReponse = addMsg('bot', html);
    scrollBas();
  }

  function renderAnswer(data) {
    majMemoire(data.memory);
    afficherReponse(data);

    // La lecture des morceaux est déjà lancée : elle se termine d'elle-même et
    // ramènera la borne au repos (cf. finLecture). Le critère est « un morceau
    // est arrivé par le flux », pas « il y en a plusieurs » : une réponse
    // courte tient en un seul morceau, et il est déjà en train d'être lu.
    if (lastSource === 'voice' && diffusionRecue && !lectureAbandonnee) return;

    // Le navigateur a refusé la lecture en cours de diffusion : le bouton
    // propose la réponse entière, dont l'URL n'était pas connue plus tôt.
    if (lectureEchouee && data.audio_url) { offerManualPlay(data.audio_url); return; }

    // Une question posée à la voix reçoit une réponse à la voix ; la session
    // ne se termine qu'une fois la lecture achevée (cf. finLecture()).
    if (lastSource === 'voice' && data.audio_url && !lectureAbandonnee) speak(data.audio_url);
    else {
      progresVers(100);
      setTimeout(progresCacher, 500);
      setState(S.IDLE);
      if (maybeEndSession()) return;
      remainingStatus('🎙️ <b>WOLOF</b> / <b>FRANÇAIS</b> pour reparler · ⌨️ ou tapez');
    }
  }

  // La remise à zéro est annoncée : sans cela, l'usager ne comprendrait pas
  // que la borne « oublie » brusquement le fil de l'échange.
  function majMemoire(mem) {
    if (!mem) return;
    memCount = mem.count;
    if (mem.max) memMax = mem.max;
    if (mem.reset) {
      const d = document.createElement('div');
      d.className = 'sysmsg';
      d.innerHTML = '🧠 Mémoire de conversation réinitialisée (' + memMax + ' messages atteints)';
      // L'avis se place AVANT la réponse. Sur le chemin vocal, la bulle existe
      // déjà quand cet avis arrive : le texte est affiché dès l'annonce de la
      // synthèse, bien avant le résultat complet qui porte l'état mémoire. Un
      // simple append le rejetterait sous la réponse, où il se lirait comme un
      // commentaire de celle-ci plutôt que comme une coupure dans le fil.
      if (bulleReponse && bulleReponse.parentNode === convo) {
        convo.insertBefore(d, bulleReponse);
      } else {
        convo.appendChild(d);
      }
      scrollBas();
    }
    renderCounter();
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Lecture de la réponse
  //
  //  La synthèse est le poste le plus lent du pipeline — jusqu'à 123 s pour
  //  61 s d'audio sur une réponse longue. Le serveur l'envoie donc par
  //  morceaux : le premier arrive vers 23 s et la borne le lit pendant que la
  //  suite se fabrique.
  //
  //  La lecture peut rattraper la synthèse (elle produit environ deux fois
  //  moins vite qu'on n'écoute) : la file se vide alors sans que la réponse
  //  soit finie. On reste en état SPEAKING, silencieux, jusqu'au morceau
  //  suivant — repasser au repos rendrait la main à l'usager en plein milieu
  //  d'une phrase.
  // ═══════════════════════════════════════════════════════════════════
  function enfilerMorceau(url, total) {
    if (!url || lectureAbandonnee) return;
    morceauxTotal = total || morceauxTotal || 1;
    fileMorceaux.push(url);
    if (!speaking) lireSuivant();
  }

  function lireSuivant() {
    if (lectureAbandonnee) return;
    const url = fileMorceaux.shift();

    if (!url) {
      if (morceauxTotal && morceauxLus >= morceauxTotal) finLecture();
      else speaking = false;          // en attente du morceau suivant
      return;
    }

    const p = getPlayer();
    p.onended = () => { morceauxLus++; lireSuivant(); };
    // Source illisible (fichier absent, format refusé) : `play()` rejette sur
    // la plupart des navigateurs, mais pas tous — d'où ce filet.
    p.onerror = () => { if (speaking) speakFailed({ name: 'NotSupportedError' }, url); };

    p.src = url + '?' + Date.now();
    speaking = true;
    setState(S.SPEAKING);
    progresLecture(morceauxLus, morceauxTotal);
    setStatus(morceauxTotal > 1
      ? '🔊 <b>Lecture de la réponse…</b> (' + Math.min(morceauxLus + 1, morceauxTotal) +
        '/' + morceauxTotal + ') — <b>STOP</b> pour interrompre'
      : '🔊 <b>Lecture de la réponse…</b> — <b>STOP</b> pour interrompre');

    p.play()
      .then(() => { unlocked = true; })
      .catch(e => speakFailed(e, url));
  }

  function finLecture() {
    speaking = false;
    progresCacher();
    setState(S.IDLE);
    if (maybeEndSession()) return;
    remainingStatus('🎙️ <b>WOLOF</b> / <b>FRANÇAIS</b> pour reparler · ⌨️ ou tapez');
  }

  // Lecture d'un fichier unique (réponse non diffusée, ou bouton ▶).
  function speak(url) {
    stopSpeaking();
    lectureAbandonnee = false;
    fileMorceaux  = [];
    morceauxLus   = 0;
    morceauxTotal = 1;
    enfilerMorceau(url, 1);
  }

  function speakFailed(err, url) {
    speaking = false;
    lectureAbandonnee = true;
    fileMorceaux = [];
    progresCacher();
    setState(S.IDLE);
    const blocked = err && err.name === 'NotAllowedError';
    // En diffusion, le bouton doit proposer la réponse entière et non le
    // morceau en cours : il est posé à l'arrivée de `result`, qui porte l'URL
    // du fichier complet.
    if (diffusionRecue) lectureEchouee = true;
    else offerManualPlay(url);
    if (maybeEndSession()) return;
    setStatus(blocked
      ? '🔇 Lecture bloquée par le navigateur — appuyez sur <b>▶ Écouter</b>'
      : '🔇 Audio indisponible — la réponse reste affichée à l’écran');
  }

  // Bouton dans la bulle : le clic est un geste utilisateur, donc jamais refusé.
  function offerManualPlay(url) {
    const bubbles = convo.querySelectorAll('.msg.bot');
    const target  = bubbles[bubbles.length - 1];
    if (!target || target.querySelector('.play-btn')) return;
    const btn = document.createElement('button');
    btn.className = 'play-btn';
    btn.textContent = '▶ Écouter la réponse';
    btn.addEventListener('click', () => {
      unlocked = true;
      btn.remove();
      beep(1100, 40);
      speak(url);
    });
    target.appendChild(btn);
    scrollBas();
  }

  // Traitement commun des événements de progression, quel que soit le canal
  // (clavier ou voix) : étiquette, jauge, affichage anticipé de la réponse et
  // mise en lecture des morceaux dès qu'ils arrivent.
  function onStatus(d) {
    if (d.step === 'tts' && (d.response || d.response_fr)) {
      // Le texte est connu avant l'audio : on l'affiche maintenant, sinon la
      // borne parlerait une minute avant de montrer ce qu'elle dit.
      removeThinking();
      afficherReponse(d);
    }
    if (d.step === 'tts_chunk' && lastSource === 'voice' && d.audio_url) {
      diffusionRecue = true;
      enfilerMorceau(d.audio_url, d.total);
    }
    updateThinking(d.step, d);
  }

  function handleError(msg) {
    removeThinking();
    progresCacher();
    addMsg('bot err', '❌ ' + esc(msg));
    if (msgCount > 0) msgCount--;          // échec : quota non consommé
    setState(S.IDLE);
    setStatus('❌ Erreur — réessayez');
  }

  // Une nouvelle question repart d'une diffusion vierge : les morceaux de la
  // précédente ne doivent jamais se glisser dans la suivante.
  function raz_diffusion() {
    fileMorceaux  = [];
    morceauxLus   = 0;
    morceauxTotal = 0;
    lectureAbandonnee = false;
    lectureEchouee    = false;
    diffusionRecue    = false;
  }

  async function sendText(text) {
    msgCount++;                   // une question = un message de la session
    raz_diffusion();
    lastSource = 'text';
    addMsg('user', esc(text));
    addThinking();
    setState(S.PROCESSING);
    abortCtl = new AbortController();
    try {
      const res = await fetch('/ask', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        // Pas d'indice de langue au clavier : le pipeline la détecte sur le texte.
        body:    JSON.stringify({ question: text, tts: false, session_id: sessionId }),
        signal:  abortCtl.signal,
      });
      await consumeSSE(res, {
        onStatus,
        onResult: d => { removeThinking(); renderAnswer(d); },
        onError:  handleError,
      });
    } catch (e) {
      if (e.name === 'AbortError') return;      // arrêt volontaire (bouton STOP)
      handleError('Réseau : ' + e.message);
    } finally {
      abortCtl = null;
      if (state === S.PROCESSING) { setState(S.IDLE); setStatus('Prêt'); }
    }
  }

  async function sendAudio(file, lang) {
    msgCount++;                   // une question = un message de la session
    raz_diffusion();
    lastSource = 'voice';
    addThinking();
    setState(S.PROCESSING);
    const fd = new FormData();
    fd.append('file', file);
    fd.append('tts',  'true');    // question vocale → réponse lue à voix haute
    fd.append('lang', lang || lastLang || 'wo');
    if (sessionId) fd.append('session_id', sessionId);
    abortCtl = new AbortController();
    try {
      const res = await fetch('/ask/audio', { method: 'POST', body: fd, signal: abortCtl.signal });
      await consumeSSE(res, {
        onStatus: d => {
          // Le pipeline renvoie la transcription dans l'événement `start`.
          if (d.step === 'start' && d.question) {
            const th = $('thinking');
            const drapeau = (LANGS[lang] || LANGS.wo).drapeau;
            const bubble = addMsg('user', '🎙️ ' + drapeau + ' ' + esc(d.question));
            if (th) convo.insertBefore(bubble, th);
            scrollBas();
          }
          onStatus(d);
        },
        onResult: d => { removeThinking(); renderAnswer(d); },
        onError:  handleError,
      });
    } catch (e) {
      if (e.name === 'AbortError') return;
      handleError('Réseau : ' + e.message);
    } finally {
      abortCtl = null;
      if (state === S.PROCESSING) { setState(S.IDLE); setStatus('Prêt'); }
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  //  Initialisation
  // ═══════════════════════════════════════════════════════════════════
  function init(opts) {
    hooks = Object.assign(hooks, opts || {});

    screen   = $('screen');   standby = $('standby'); session = $('session');
    convo    = $('convo');    statusEl = $('status'); recTime = $('rec-time');
    counter  = $('counter');  keyboard = $('keyboard'); kbInput = $('kb-input');
    memEl    = $('mem');
    vizEl    = $('viz');      toastEl  = $('toast');
    progresInit();            // jauge ajoutée à la barre d'état, si présente

    for (let i = 0; i < VIZ_BARS; i++) vizEl.appendChild(document.createElement('i'));
    buildKeyboard();

    // Raccourcis clavier : simulation des appuis physiques.
    document.addEventListener('keydown', e => {
      if (e.target === kbInput && e.key !== 'Escape') {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); pressStartStop(); }
        return;
      }
      if (e.repeat) return;
      if (e.key === '1') { hooks.onKey && hooks.onKey('wo', true); pressLang('wo'); }
      else if (e.key === '2') { hooks.onKey && hooks.onKey('fr', true); pressLang('fr'); }
      else if (e.key === '3' || e.key === ' ') {
        e.preventDefault();
        hooks.onKey && hooks.onKey('power', true);
        holdStart();
      } else if (e.key === 'Escape') powerOff();
    });

    document.addEventListener('keyup', e => {
      if (e.key === '3' || e.key === ' ') holdEnd(true);
      ['wo', 'fr', 'power'].forEach(k => hooks.onKey && hooks.onKey(k, false));
    });

    // État du service
    (async () => {
      let d = null;
      try { d = await (await fetch('/health')).json(); } catch (_) { }
      if (hooks.onHealth) hooks.onHealth(d);
    })();

    render();
  }

  return {
    init, info, pressLang, pressStartStop, holdStart, holdEnd, powerOn, powerOff,
    speak, unlockAudio, toast, beep, S, LANGS, MAX_MESSAGES, IDLE_TIMEOUT,
  };
})();
