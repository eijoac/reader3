// Keep the current chapter's TOC entry in view
const activeLink = document.querySelector('a.toc-link.active');
if (activeLink) activeLink.scrollIntoView({ block: 'center' });

// Sidebar toggle logic with persistence
function toggleSidebar() {
    const isHidden = document.body.classList.toggle('sidebar-hidden');
    try { 
        localStorage.setItem('sidebarHidden', isHidden ? '1' : '0'); 
    } catch (e) {}
}
document.getElementById('toc-toggle').addEventListener('click', toggleSidebar);

// Initialize state from localStorage
(function initSidebarState() {
    try {
        const hidden = localStorage.getItem('sidebarHidden') === '1';
        if (hidden) {
            document.body.classList.add('sidebar-hidden');
        }
    } catch (e) {}
})();

// Theme toggle (toggleTheme and the initial theme come from theme.js)
document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

// Reading font and size. Changing the size reflows the page, so keep the reader at the same place in the text.
const SCALE_MIN = 0.8, SCALE_MAX = 1.8, SCALE_STEP = 0.1;
const rootEl = document.documentElement;
const smallerBtn = document.getElementById('font-smaller');
const largerBtn = document.getElementById('font-larger');

function currentScale() {
    return parseFloat(getComputedStyle(rootEl).getPropertyValue('--font-scale')) || 1;
}

function setScale(value) {
    const scale = Math.round(Math.min(SCALE_MAX, Math.max(SCALE_MIN, value)) * 10) / 10;
    const mainEl = document.getElementById('main');
    const range = mainEl.scrollHeight - mainEl.clientHeight;
    const frac = range > 0 ? mainEl.scrollTop / range : 0;
    rootEl.style.setProperty('--font-scale', scale);
    const newRange = mainEl.scrollHeight - mainEl.clientHeight;
    mainEl.scrollTo({ top: frac * newRange, behavior: 'instant' });
    try { localStorage.setItem('reader3:scale', String(scale)); } catch (e) {}
    updateFontControls();
}

function updateFontControls() {
    const scale = currentScale();
    smallerBtn.disabled = scale <= SCALE_MIN + 0.001;
    largerBtn.disabled = scale >= SCALE_MAX - 0.001;
}

function toggleFont() {
    const next = rootEl.dataset.font === 'sans' ? 'serif' : 'sans';
    if (next === 'sans') rootEl.dataset.font = 'sans'; else delete rootEl.dataset.font;
    try { localStorage.setItem('reader3:font', next); } catch (e) {}
}

smallerBtn.addEventListener('click', () => setScale(currentScale() - SCALE_STEP));
largerBtn.addEventListener('click', () => setScale(currentScale() + SCALE_STEP));
document.getElementById('font-toggle').addEventListener('click', toggleFont);
updateFontControls();

// Reading position: remember the last chapter and how far down it the reader was
// Per-page data comes from a JSON block in reader.html (not executed, so a strict CSP allows it)
const { bookId: BOOK_ID, chapter: CHAPTER, prevUrl: PREV_URL, nextUrl: NEXT_URL } =
    JSON.parse(document.getElementById('reader-config').textContent);
const POS_KEY = 'reader3:pos:' + BOOK_ID;
const main = document.getElementById('main');

function readPosition() {
    try {
        const pos = JSON.parse(localStorage.getItem(POS_KEY));
        if (pos && Number.isInteger(pos.chapter) && typeof pos.frac === 'number') return pos;
    } catch (e) {}
    return null;
}

function scrollFraction() {
    const range = main.scrollHeight - main.clientHeight;
    return range > 0 ? Math.min(1, Math.max(0, main.scrollTop / range)) : 0;
}

function savePosition() {
    try {
        localStorage.setItem(POS_KEY, JSON.stringify({ chapter: CHAPTER, frac: scrollFraction() }));
    } catch (e) {}
}

function restoreScroll(frac) {
    const range = main.scrollHeight - main.clientHeight;
    // "instant" bypasses the smooth-scrolling CSS so the page doesn't visibly glide down
    main.scrollTo({ top: frac * range, behavior: 'instant' });
}

(function initReadingPosition() {
    const saved = readPosition();
    // Only resume the same chapter, and never fight an in-page #anchor link
    const resume = saved && saved.chapter === CHAPTER && !location.hash ? saved.frac : null;
    let interacted = false;
    ['wheel', 'touchstart', 'keydown', 'mousedown'].forEach(evt =>
        main.addEventListener(evt, () => { interacted = true; }, { passive: true }));

    if (resume !== null) {
        restoreScroll(resume);
        // Images load after first layout and change the page height; apply the same fraction again
        window.addEventListener('load', () => { if (!interacted) restoreScroll(resume); });
    } else {
        savePosition();
    }

    let timer = null;
    main.addEventListener('scroll', () => {
        clearTimeout(timer);
        timer = setTimeout(savePosition, 250);
    }, { passive: true });
    document.addEventListener('visibilitychange', () => { if (document.hidden) savePosition(); });
    window.addEventListener('pagehide', savePosition);
})();

// Keyboard: ←/→ change chapter, t contents, d dark mode, f font, +/- text size
document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.closest && e.target.closest('input, textarea, select, [contenteditable]')) return;
    switch (e.key) {
        case 'ArrowLeft':  if (PREV_URL) location.href = PREV_URL; break;
        case 'ArrowRight': if (NEXT_URL) location.href = NEXT_URL; break;
        case 't': case 'T': toggleSidebar(); break;
        case 'd': case 'D': toggleTheme(); break;
        case 'f': case 'F': toggleFont(); break;
        case '+': case '=': setScale(currentScale() + SCALE_STEP); break;
        case '-': case '_': setScale(currentScale() - SCALE_STEP); break;
        default: return;
    }
    e.preventDefault();
});

// Focus the reading area so arrow keys, space and PageUp/PageDown scroll it right away
main.focus({ preventScroll: true });
