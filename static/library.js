document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

// Resume where the reader left off (position is saved by the reader page)
document.querySelectorAll('a.read-btn').forEach((btn) => {
    try {
        const pos = JSON.parse(localStorage.getItem('reader3:pos:' + btn.dataset.bookId));
        const total = parseInt(btn.dataset.chapters, 10);
        if (pos && Number.isInteger(pos.chapter) && pos.chapter >= 0 && pos.chapter < total
            && (pos.chapter > 0 || pos.frac > 0.01)) {
            btn.href = '/read/' + encodeURIComponent(btn.dataset.bookId) + '/' + pos.chapter;
            btn.textContent = 'Continue (section ' + (pos.chapter + 1) + ')';
        }
    } catch (e) {}
});
