document.addEventListener("DOMContentLoaded", function () {
    const highlightButton = document.getElementById("highlightButton");
  
    highlightButton.addEventListener("click", function () {
      chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        chrome.scripting.executeScript({
          target: { tabId: tabs[0].id },
          function: highlightText,
        });
      });
    });
  });
  
  function highlightText() {
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
  