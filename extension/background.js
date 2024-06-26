chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "highlightText") {
        browser.tabs.executeScript(sender.tab.id, {
            code: `(${highlightText.toString()})();`
        }).then(sendResponse);
        return true;  // Indicates the response is sent asynchronously
    }
});

function highlightText() {
    // This will be injected into the content script.
    const selectedText = window.getSelection().toString();
    if (selectedText) {
        const span = document.createElement("span");
        span.style.backgroundColor = "yellow";
        span.textContent = selectedText;

        const range = window.getSelection().getRangeAt(0);
        range.deleteContents();
        range.insertNode(span);
    }
}
