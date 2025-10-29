/*
  Enhanced app.js for video + subtitle upload & playback
  - Preserves existing core functionality (video + subtitle upload and playback)
  - Adds drag-and-drop, clear status messages, loading indicators, validation, accessibility
  - Minimal, framework-agnostic styling hooks (works with Tailwind/Bootstrap if present)
  - Modular functions and descriptive comments

  Assumptions: The HTML includes at least
    - A <video id="videoPlayer" controls></video>
    - An <input type="file" id="videoInput" accept="video/*">
    - An <input type="file" id="subtitleInput" accept=".vtt,.srt,.sub,.sbv,.dfxp,.ttml,text/vtt">

  Optional (not required): If you add elements with these IDs, the script will enhance UX automatically:
    - #videoDropZone and #subtitleDropZone (drop targets that can contain the file inputs)
    - A status area (#statusArea) and spinner (#loadingSpinner) are auto-created if not found.

  Note: We only modify this app.js file as requested. Minimal CSS is injected dynamically for better UX without changing other files.
*/

(function () {
  'use strict';

  // --------- Configuration & selectors ---------
  const SELECTORS = {
    video: '#videoPlayer',
    videoInput: '#videoInput',
    subtitleInput: '#subtitleInput',
    // Optional containers (auto-created if missing)
    statusAreaId: 'statusArea',
    spinnerId: 'loadingSpinner',
    videoDropId: 'videoDropZone',
    subtitleDropId: 'subtitleDropZone'
  };

  // Accessible text helpers
  const ariaLivePolite = { role: 'status', 'aria-live': 'polite' };

  // --------- DOM references ---------
  let $video, $videoInput, $subtitleInput, $status, $spinner, $videoDrop, $subtitleDrop;

  // --------- State ---------
  const state = {
    videoURL: null,
    subtitleURL: null,
    activeTrack: null,
  };

  // --------- Utilities ---------
  function $(selector) {
    return document.querySelector(selector);
  }

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

  function setAttrs(el, attrs) {
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  }

  function announce(message) {
    if (!$status) return;
    $status.textContent = '';
    requestAnimationFrame(() => {
      $status.textContent = message;
    });
    $status.classList.remove('opacity-0');
    $status.classList.add('opacity-100');
  }

  function showSpinner(msg = 'Processing...') {
    if (!$spinner) return;
    $spinner.style.display = 'inline-flex';
    $spinner.setAttribute('aria-label', msg);
  }

  function hideSpinner() {
    if (!$spinner) return;
    $spinner.style.display = 'none';
  }

  function revokeURL(url) {
    try { url && URL.revokeObjectURL(url); } catch (_) { /* noop */ }
  }

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

  function fileIsVideo(file) {
    if (!file) return false;
    const type = (file.type || '').toLowerCase();
    return type.startsWith('video/');
  }

  // --------- Video & Subtitle handling ---------
  function attachSubtitleTrack(videoEl, url, label = 'Subtitles', lang = 'en') {
    // Remove existing tracks
    Array.from(videoEl.querySelectorAll('track')).forEach(t => t.remove());

    const track = document.createElement('track');
    track.kind = 'subtitles';
    track.label = label;
    track.srclang = lang;
    track.src = url;
    track.default = true;
    videoEl.appendChild(track);
    state.activeTrack = track;

    track.addEventListener('load', () => {
      announce('Subtitle loaded successfully.');
    });

    track.addEventListener('error', () => {
      announce('Failed to load subtitle. Please check the file format.');
    });
  }

  function loadVideo(file) {
    if (!file) {
      announce('No video file selected.');
      return;
    }
    if (!fileIsVideo(file)) {
      announce('Unsupported video type. Please choose a valid video file.');
      return;
    }

    showSpinner('Loading video...');
    revokeURL(state.videoURL);
    state.videoURL = URL.createObjectURL(file);
    $video.src = state.videoURL;

    $video.onloadeddata = () => {
      hideSpinner();
      $video.classList.add('fade-in');
      announce('Video ready to play.');
      $video.focus();
    };

    $video.onerror = () => {
      hideSpinner();
      announce('Error loading video file.');
    };
  }

  function loadSubtitle(file) {
    if (!file) {
      announce('No subtitle file selected.');
      return;
    }
    if (!fileIsSubtitle(file)) {
      announce('Unsupported subtitle type. Use .vtt or .srt if possible.');
      return;
    }

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
        } catch (e) {
          hideSpinner();
          announce('Failed to parse SRT file.');
        }
      };
      reader.onerror = () => {
        hideSpinner();
        announce('Error reading subtitle file.');
      };
      reader.readAsText(file);
    } else {
      state.subtitleURL = URL.createObjectURL(file);
      attachSubtitleTrack($video, state.subtitleURL);
      hideSpinner();
      announce('Subtitle loaded successfully.');
    }
  }

  // Simple SRT -> VTT converter (basic, handles common cases)
  function srtToVtt(srt) {
    let vtt = 'WEBVTT\n\n';
    const lines = srt.replace(/\r/g, '').split('\n');
    for (let i = 0; i < lines.length; i++) {
      let line = lines[i];
      // Convert time format 00:00:01,000 --> 00:00:01.000
      line = line.replace(/(\d\d:\d\d:\d\d),(\d\d\d)/g, '$1.$2');
      // Skip numeric cue identifiers (optional)
      if (/^\d+$/.test(line.trim())) continue;
      vtt += line + '\n';
    }
    return vtt;
  }

  // --------- Drag & Drop helpers ---------
  function makeDropZone(el, { onFiles, label }) {
    if (!el) return;

    el.setAttribute('tabindex', '0');
    el.setAttribute('role', 'button');
    el.setAttribute('aria-label', label);

    const onDragOver = (e) => {
      e.preventDefault();
      el.classList.add('dropzone-hover');
    };
    const onDragLeave = (e) => {
      e.preventDefault();
      el.classList.remove('dropzone-hover');
    };
    const onDrop = (e) => {
      e.preventDefault();
      el.classList.remove('dropzone-hover');
      const files = Array.from(e.dataTransfer.files || []);
      if (files.length === 0) return announce('No file dropped.');
      onFiles(files);
    };

    el.addEventListener('dragover', onDragOver);
    el.addEventListener('dragleave', onDragLeave);
    el.addEventListener('drop', onDrop);

    // Keyboard accessible: Enter/Space to trigger underlying input click if present
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        const input = el.querySelector('input[type="file"]');
        if (input) input.click();
      }
    });
  }

  // --------- Event wiring ---------
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

    // Drag & drop zones (if present)
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

    // Improve keyboard focus outlines if user navigates by keyboard
    function handleFirstTab(e) {
      if (e.key === 'Tab') {
        document.body.classList.add('user-is-tabbing');
        window.removeEventListener('keydown', handleFirstTab);
      }
    }
    window.addEventListener('keydown', handleFirstTab);
  }

  // --------- Initialization ---------
  function init() {
    $video = $(SELECTORS.video);
    $videoInput = $(SELECTORS.videoInput);
    $subtitleInput = $(SELECTORS.subtitleInput);

    // Create or find status area (ARIA live region)
    $status = ensureContainer(SELECTORS.statusAreaId, {
      classes: 'status-area transition-opacity duration-200 ease-out opacity-100',
      attrs: ariaLivePolite
    });

    // Create spinner (hidden by default)
    $spinner = ensureContainer(SELECTORS.spinnerId, {
      classes: 'spinner items-center gap-2',
      attrs: { 'aria-hidden': 'true' }
    });
    $spinner.style.display = 'none';
    $spinner.innerHTML = `
      <span class="spinner-dot" aria-hidden="true"></span>
      <span class="spinner-text">Loading...</span>
    `;

    // Optional drop zones if present in DOM
    $videoDrop = document.getElementById(SELECTORS.videoDropId) || null;
    $subtitleDrop = document.getElementById(SELECTORS.subtitleDropId) || null;

    // If no video element, create a basic one (non-breaking if already exists)
    if (!$video) {
      $video = document.createElement('video');
      $video.id = 'videoPlayer';
      $video.controls = true;
      $video.className = 'video-player';
      document.body.appendChild($video);
    }

    // Accessibility: focus outline support
    $video.setAttribute('tabindex', '0');

    bindEvents();
    announce('Ready. Choose or drop a video file and a subtitle file.');
  }

  // --------- Minimal CSS hooks (optional, injected) ---------
  function injectStyles() {
    const css = `
      /* Visually distinct status area */
      .status-area { font-size: 0.95rem; color: #1f2937; margin: 0.5rem 0; }

      /* Spinner */
      .spinner { position: fixed; top: 1rem; right: 1rem; background: rgba(255,255,255,0.9); border-radius: 0.5rem; padding: 0.5rem 0.75rem; box-shadow: 0 4px 18px rgba(0,0,0,0.1); z-index: 9999; }
      .spinner-dot { width: 10px; height: 10px; border-radius: 50%; background: #2563EB; display: inline-block; animation: pulse 1s infinite; }
      .spinner-text { margin-left: 0.5rem; color: #111827; }
      @keyframes pulse { 0%{opacity:.3} 50%{opacity:1} 100%{opacity:.3} }

      /* Drop zones */
      .dropzone { border: 2px dashed #cbd5e1; border-radius: 0.5rem; padding: 1rem; text-align: center; color: #475569; transition: background-color .2s ease, border-color .2s ease, transform .15s ease; }
      .dropzone-hover { background-color: #f1f5f9; border-color: #2563EB; transform: translateY(-1px); }

      /* Video */
      .video-player { width: 100%; max-width: 960px; display: block; margin: 1rem auto; border-radius: 0.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.08); opacity: 0; }
      .fade-in { animation: fadeIn .35s ease-out forwards; }
      @keyframes fadeIn { to { opacity: 1; } }

      /* Focus styles for accessibility */
      .user-is-tabbing :focus { outline: 3px solid #2563EB !important; outline-offset: 2px; border-radius: 0.25rem; }
    `;
    const styleEl = document.createElement('style');
    styleEl.setAttribute('data-appjs-enhancements', '');
    styleEl.appendChild(document.createTextNode(css));
    document.head.appendChild(styleEl);
  }

  // --------- Run ---------
  document.addEventListener('DOMContentLoaded', () => {
    injectStyles();
    init();
  });
})();

/*
Optional HTML guidance (do not modify other files unless desired):

<div id="videoDropZone" class="dropzone" aria-describedby="videoHelp">
  <p><strong>Video</strong>: Drag & drop here or press Enter to browse</p>
  <input id="videoInput" type="file" accept="video/*" style="display:none" />
</div>
<small id="videoHelp">Supported: common video formats (mp4, webm, etc.).</small>

<div id="subtitleDropZone" class="dropzone" aria-describedby="subtitleHelp">
  <p><strong>Subtitles</strong>: Drag & drop here or press Enter to browse</p>
  <input id="subtitleInput" type="file" accept=".vtt,.srt,.sub,.sbv,.dfxp,.ttml,text/vtt" style="display:none" />
</div>
<small id="subtitleHelp">Recommended: .vtt. .srt is supported and auto-converted.</small>

<video id="videoPlayer" controls aria-label="Video player with subtitles"></video>
*/
