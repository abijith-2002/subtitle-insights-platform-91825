(function () {
  'use strict';

  const SELECTORS = {
    video: '#videoPlayer',
    videoInput: '#videoInput',
    subtitleInput: '#subtitleInput',
    statusAreaId: 'statusArea',
    spinnerId: 'loadingSpinner',
    videoDropId: 'videoDropZone',
    subtitleDropId: 'subtitleDropZone'
  };

  let $video, $videoInput, $subtitleInput, $status, $spinner, $videoDrop, $subtitleDrop;

  const state = {
    videoURL: null,
    subtitleURL: null,
    activeTrack: null
  };

  function $(selector) { return document.querySelector(selector); }

  function ensureContainer(id, { classes = '', attrs = {} } = {}) {
    let el = document.getElementById(id);
    if (!el) {
      el = document.createElement('div');
      el.id = id;
      if (classes) el.className = classes;
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      document.body.appendChild(el);
    }
    return el;
  }

  function announce(message) {
    if (!$status) return;
    $status.textContent = '';
    requestAnimationFrame(() => { $status.textContent = message; });
    $status.classList.remove('opacity-0');
    $status.classList.add('opacity-100');
  }

  function showSpinner(msg = 'Processing...') {
    if (!$spinner) return;
    $spinner.style.display = 'inline-flex';
    $spinner.setAttribute('aria-label', msg);
  }
  function hideSpinner() { if ($spinner) $spinner.style.display = 'none'; }
  function revokeURL(url) { try { url && URL.revokeObjectURL(url); } catch (_) {} }

  function fileIsSubtitle(file) {
    if (!file) return false;
    const name = (file.name || '').toLowerCase();
    const type = (file.type || '').toLowerCase();
    return (
      name.endsWith('.vtt') || name.endsWith('.srt') || name.endsWith('.sub') ||
      name.endsWith('.sbv') || name.endsWith('.dfxp') || name.endsWith('.ttml') ||
      type === 'text/vtt' || type.includes('subtitle')
    );
  }
  function fileIsVideo(file) { return !!(file && (file.type || '').toLowerCase().startsWith('video/')); }

  function attachSubtitleTrack(videoEl, url, label = 'Subtitles', lang = 'en') {
    Array.from(videoEl.querySelectorAll('track')).forEach(t => t.remove());
    const track = document.createElement('track');
    track.kind = 'subtitles';
    track.label = label;
    track.srclang = lang;
    track.src = url;
    track.default = true;
    videoEl.appendChild(track);
    state.activeTrack = track;

    track.addEventListener('load', () => announce('Subtitle loaded successfully.'));
    track.addEventListener('error', () => announce('Failed to load subtitle. Please check the file format.'));
  }

  function loadVideo(file) {
    if (!file) return announce('No video file selected.');
    if (!fileIsVideo(file)) return announce('Unsupported video type. Please choose a valid video file.');
    showSpinner('Loading video...');
    revokeURL(state.videoURL);
    state.videoURL = URL.createObjectURL(file);
    $video.src = state.videoURL;
    $video.onloadeddata = () => { hideSpinner(); $video.classList.add('fade-in'); announce('Video ready to play.'); $video.focus(); };
    $video.onerror = () => { hideSpinner(); announce('Error loading video file.'); };
  }

  function loadSubtitle(file) {
    if (!file) return announce('No subtitle file selected.');
    if (!fileIsSubtitle(file)) return announce('Unsupported subtitle type. Use .vtt or .srt if possible.');
    showSpinner('Loading subtitles...');
    revokeURL(state.subtitleURL);
    const name = (file.name || '').toLowerCase();
    if (name.endsWith('.srt')) {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const vttText = srtToVtt(String(reader.result));
          const blob = new Blob([vttText], { type: 'text/vtt' });
          state.subtitleURL = URL.createObjectURL(blob);
          attachSubtitleTrack($video, state.subtitleURL);
          hideSpinner();
          announce('Subtitle (SRT) converted to VTT and loaded.');
        } catch (e) { hideSpinner(); announce('Failed to parse SRT file.'); }
      };
      reader.onerror = () => { hideSpinner(); announce('Error reading subtitle file.'); };
      reader.readAsText(file);
    } else {
      state.subtitleURL = URL.createObjectURL(file);
      attachSubtitleTrack($video, state.subtitleURL);
      hideSpinner();
      announce('Subtitle loaded successfully.');
    }
  }

  function srtToVtt(srt) {
    let vtt = 'WEBVTT\n\n';
    const lines = srt.replace(/\r/g, '').split('\n');
    for (let i = 0; i < lines.length; i++) {
      let line = lines[i];
      line = line.replace(/(\d\d:\d\d:\d\d),(\d\d\d)/g, '$1.$2');
      if (/^\d+$/.test(line.trim())) continue;
      vtt += line + '\n';
    }
    return vtt;
  }

  function makeDropZone(el, { onFiles, label }) {
    if (!el) return;
    el.setAttribute('tabindex', '0');
    el.setAttribute('role', 'button');
    el.setAttribute('aria-label', label);

    el.addEventListener('dragover', (e) => { e.preventDefault(); el.classList.add('dropzone-hover'); });
    el.addEventListener('dragleave', (e) => { e.preventDefault(); el.classList.remove('dropzone-hover'); });
    el.addEventListener('drop', (e) => {
      e.preventDefault();
      el.classList.remove('dropzone-hover');
      const files = Array.from(e.dataTransfer.files || []);
      if (files.length === 0) return announce('No file dropped.');
      onFiles(files);
    });
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        const input = el.querySelector('input[type="file"]');
        if (input) input.click();
      }
    });
  }

  function bindEvents() {
    if ($videoInput) {
      $videoInput.addEventListener('change', (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return announce('No video file selected.');
        loadVideo(file);
      });
    }
    if ($subtitleInput) {
      $subtitleInput.addEventListener('change', (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return announce('No subtitle file selected.');
        loadSubtitle(file);
      });
    }
    if ($videoDrop) {
      makeDropZone($videoDrop, {
        label: 'Drop video file here or press Enter to browse',
        onFiles: (files) => {
          const file = files.find(fileIsVideo) || files[0];
          if (!fileIsVideo(file)) return announce('Please drop a valid video file.');
          loadVideo(file);
        }
      });
    }
    if ($subtitleDrop) {
      makeDropZone($subtitleDrop, {
        label: 'Drop subtitle file here or press Enter to browse',
        onFiles: (files) => {
          const file = files.find(fileIsSubtitle) || files[0];
          if (!fileIsSubtitle(file)) return announce('Please drop a valid subtitle file (.vtt or .srt).');
          loadSubtitle(file);
        }
      });
    }
    function handleFirstTab(e) {
      if (e.key === 'Tab') {
        document.body.classList.add('user-is-tabbing');
        window.removeEventListener('keydown', handleFirstTab);
      }
    }
    window.addEventListener('keydown', handleFirstTab);
  }

  function init() {
    $video = $(SELECTORS.video);
    $videoInput = $(SELECTORS.videoInput);
    $subtitleInput = $(SELECTORS.subtitleInput);

    $status = ensureContainer(SELECTORS.statusAreaId, {
      classes: 'status-area transition-opacity duration-200 ease-out opacity-100',
      attrs: { role: 'status', 'aria-live': 'polite' }
    });

    $spinner = ensureContainer(SELECTORS.spinnerId, {
      classes: 'spinner items-center gap-2',
      attrs: { 'aria-hidden': 'true' }
    });
    $spinner.style.display = 'none';
    $spinner.innerHTML = '<span class="spinner-dot" aria-hidden="true"></span><span class="spinner-text">Loading...</span>';

    $videoDrop = document.getElementById(SELECTORS.videoDropId) || null;
    $subtitleDrop = document.getElementById(SELECTORS.subtitleDropId) || null;

    if (!$video) {
      $video = document.createElement('video');
      $video.id = 'videoPlayer';
      $video.controls = true;
      $video.className = 'video-player';
      document.body.appendChild($video);
    }
    $video.setAttribute('tabindex', '0');

    bindEvents();
    announce('Ready. Choose or drop a video file and a subtitle file.');
  }

  function injectStyles() {
    const css = `
      .status-area { font-size: 0.95rem; color: #1f2937; margin: 0.5rem 0; }
      .spinner { position: fixed; top: 1rem; right: 1rem; background: rgba(255,255,255,0.9); border-radius: 0.5rem; padding: 0.5rem 0.75rem; box-shadow: 0 4px 18px rgba(0,0,0,0.1); }
      .spinner-dot { width: 10px; height: 10px; border-radius: 50%; background: #2563EB; display: inline-block; animation: pulse 1s infinite; }
      .spinner-text { margin-left: 0.5rem; color: #111827; }
      @keyframes pulse { 0%{opacity:.3} 50%{opacity:1} 100%{opacity:.3} }
      .dropzone { border: 2px dashed #cbd5e1; border-radius: 0.5rem; padding: 1rem; text-align: center; color: #475569; transition: background-color .2s ease, border-color .2s ease, transform .15s ease; }
      .dropzone-hover { background-color: #f1f5f9; border-color: #2563EB; transform: translateY(-1px); }
      .video-player { width: 100%; max-width: 960px; display: block; margin: 1rem auto; border-radius: 0.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.08); opacity: 0; }
      .fade-in { animation: fadeIn .35s ease-out forwards; }
      @keyframes fadeIn { to { opacity: 1; } }
      .user-is-tabbing :focus { outline: 3px solid #2563EB !important; outline-offset: 2px; border-radius: 0.25rem; }
    `;
    const styleEl = document.createElement('style');
    styleEl.setAttribute('data-appjs-enhancements', '');
    styleEl.appendChild(document.createTextNode(css));
    document.head.appendChild(styleEl);
  }

  document.addEventListener('DOMContentLoaded', () => { injectStyles(); init(); });
})();
