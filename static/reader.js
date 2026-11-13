// Keep the current chapter's TOC entry in view
const activeLink = document.querySelector('a.toc-link.active');
if (activeLink) activeLink.scrollIntoView({ block: 'center' });

// Sidebar toggle logic with persistence
const sidebar = document.getElementById('sidebar');
const tocToggle = document.getElementById('toc-toggle');
const main = document.getElementById('main');
const readerControls = document.querySelector('.reader-controls');
const mobileLayout = window.matchMedia('(max-width: 760px)');

function setSidebarHidden(hidden) {
    document.body.classList.toggle('sidebar-hidden', hidden);
    sidebar.inert = hidden;
    tocToggle.setAttribute('aria-expanded', String(!hidden));
    main.inert = mobileLayout.matches && !hidden;
    readerControls.inert = mobileLayout.matches && !hidden;
}

function toggleSidebar() {
    const isHidden = !document.body.classList.contains('sidebar-hidden');
    setSidebarHidden(isHidden);
    if (!isHidden && mobileLayout.matches) document.getElementById('book-search').focus();
    try {
        localStorage.setItem('reader3:sidebarHidden', isHidden ? '1' : '0');
    } catch (e) {}
}
tocToggle.addEventListener('click', toggleSidebar);
document.getElementById('sidebar-backdrop').addEventListener('click', () => {
    setSidebarHidden(true);
    tocToggle.focus();
});

// Initialize state from localStorage
(function initSidebarState() {
    try {
        // Falls back to the key used before it got the reader3: prefix
        const stored = localStorage.getItem('reader3:sidebarHidden') ?? localStorage.getItem('sidebarHidden');
        setSidebarHidden(stored === '1' || (stored === null && mobileLayout.matches));
    } catch (e) {
        setSidebarHidden(mobileLayout.matches);
    }
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

const readerConfig = JSON.parse(document.getElementById('reader-config').textContent);
const BOOK_ID = readerConfig.bookId;
const SEARCH_URL = readerConfig.searchUrl;
const searchForm = document.getElementById('book-search-form');
const searchInput = document.getElementById('book-search');
const searchStatus = document.getElementById('search-status');
const searchResults = document.getElementById('search-results');
let searchController = null;

searchForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const query = searchInput.value.trim();
    searchResults.replaceChildren();
    if (!query) {
        searchStatus.textContent = 'Enter a search term.';
        searchInput.focus();
        return;
    }

    if (searchController) searchController.abort();
    const controller = new AbortController();
    searchController = controller;
    searchForm.setAttribute('aria-busy', 'true');
    searchStatus.textContent = 'Searching...';

    try {
        const params = new URLSearchParams({ q: query });
        const response = await fetch(`${SEARCH_URL}?${params}`, { signal: controller.signal });
        if (!response.ok) throw new Error(`Search request failed with status ${response.status}`);
        const data = await response.json();

        for (const result of data.results) {
            const item = document.createElement('li');
            const link = document.createElement('a');
            link.className = 'search-result-link';
            link.href = `/read/${encodeURIComponent(BOOK_ID)}/${result.chapter}`;

            const title = document.createElement('span');
            title.className = 'search-result-title';
            title.textContent = `Section ${result.chapter + 1}: ${result.title}`;
            const snippet = document.createElement('span');
            snippet.className = 'search-result-snippet';
            snippet.textContent = result.snippet;
            link.append(title, snippet);
            item.append(link);
            searchResults.append(item);
        }

        if (data.results.length === 0) {
            searchStatus.textContent = 'No matching sections found.';
        } else if (data.truncated) {
            searchStatus.textContent = `Showing the first ${data.results.length} matching sections. Refine your search for more specific results.`;
        } else {
            searchStatus.textContent = `${data.results.length} matching section${data.results.length === 1 ? '' : 's'} found.`;
        }
    } catch (error) {
        if (error.name === 'AbortError') return;
        console.error('Book search failed', error);
        searchStatus.textContent = 'Search failed. Check your connection and try again.';
    } finally {
        if (searchController === controller) {
            searchController = null;
            searchForm.removeAttribute('aria-busy');
        }
    }
});

// Reading position: remember the last chapter and how far down it the reader was
// Per-page data comes from a JSON block in reader.html (not executed, so a strict CSP allows it)
const { chapter: CHAPTER, prevUrl: PREV_URL, nextUrl: NEXT_URL } = readerConfig;
const POS_KEY = 'reader3:pos:' + BOOK_ID;

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
    if (e.key === 'Escape' && mobileLayout.matches && !document.body.classList.contains('sidebar-hidden')) {
        setSidebarHidden(true);
        tocToggle.focus();
        e.preventDefault();
        return;
    }
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
