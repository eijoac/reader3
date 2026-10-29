// Apply the saved (or OS) theme before first paint to avoid a flash
(function () {
    let theme = null;
    try { theme = localStorage.getItem('reader3:theme'); } catch (e) {}
    if (theme !== 'light' && theme !== 'dark') {
        theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    document.documentElement.dataset.theme = theme;

    // Reading font and size (applied here too, so the page doesn't reflow after load)
    try {
        if (localStorage.getItem('reader3:font') === 'sans') document.documentElement.dataset.font = 'sans';
        const scale = parseFloat(localStorage.getItem('reader3:scale'));
        if (scale >= 0.8 && scale <= 1.8) document.documentElement.style.setProperty('--font-scale', scale);
    } catch (e) {}
})();

// Shared by both pages: flip between light and dark and remember the choice
function toggleTheme() {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('reader3:theme', next); } catch (e) {}
}
