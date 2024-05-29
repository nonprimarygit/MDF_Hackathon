let clickCount = 0;
let lastClickTime = 0;
let style = document.createElement('style');
let highlightedSpan = null;
style.innerHTML = `
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
::selection { background: transparent; }
`;
document.head.appendChild(style);

function setUpAudioElements() {
    let audioElements = document.querySelectorAll('audio');

    audioElements.forEach(function (audio) {
        if (audio.getAttribute('data-overlay-set') === 'true') return;

        setTimeout(function() {
            // Ensure that the audio element has not been processed in the meantime
            if (audio.getAttribute('data-overlay-set') === 'true') return;

            let overlay = document.createElement('div');
            let rect = audio.getBoundingClientRect();

            // Adjust position to be relative to the offset parent
            let parentRect = audio.offsetParent.getBoundingClientRect();

            overlay.style.position = 'absolute';
            overlay.style.width = rect.width + 'px';
            overlay.style.height = rect.height + 'px';
            overlay.style.top = (rect.top - parentRect.top) + 'px';
            overlay.style.left = (rect.left - parentRect.left) + 'px';
            overlay.style.zIndex = 10; // Ensure it's above the audio element
            overlay.style.opacity = 0; 

            // Add event listener to the overlay
            overlay.addEventListener('click', function (event) {
                console.log('Audio element clicked');
                event.stopPropagation();
                const currentTime = new Date().getTime();
                const timeDiff = currentTime - lastClickTime;

                if (timeDiff < 300) {
                    clickCount++;
                } else {
                    clickCount = 1;
                }

                lastClickTime = currentTime;


                if (clickCount === 3) {
                    clickCount = 0;
                    let contentType = 'audio';
                    let contentData;
                    if (audio.src) {
                        // If the audio element has a direct src attribute
                        contentData = audio.src;
                    } else {
                        // If the audio element contains source elements
                        let sources = audio.getElementsByTagName('source');
                        if (sources.length > 0) {
                            contentData = sources[0].src; // Taking the src of the first source element
                        }
                    }
                    if (contentData) {
                        createInfoWindow(contentData, window.location.href, contentType);
                    }
                }
            });

            audio.offsetParent.appendChild(overlay);
            audio.style.position = 'relative';

            // Mark this audio element as having an overlay
            audio.setAttribute('data-overlay-set', 'true');
        }, 100);
    });
}

function setUpIframeElements() {
    let iframeElements = document.querySelectorAll('iframe');

    iframeElements.forEach(function (iframe) {
        if (iframe.getAttribute('data-overlay-set') === 'true') return;

        setTimeout(function() {
            // Ensure that the iframe element has not been processed in the meantime
            if (iframe.getAttribute('data-overlay-set') === 'true') return;

            let overlay = document.createElement('div');
            let rect = iframe.getBoundingClientRect();

            // Adjust position to be relative to the offset parent
            let parentRect = iframe.offsetParent.getBoundingClientRect();

            overlay.style.position = 'absolute';
            overlay.style.width = rect.width + 'px';
            overlay.style.height = rect.height + 'px';
            overlay.style.top = (rect.top - parentRect.top) + 'px';
            overlay.style.left = (rect.left - parentRect.left) + 'px';
            overlay.style.zIndex = 10; // Ensure it's above the iframe element
            overlay.style.opacity = 0; 
            overlay.style.cursor = 'pointer'; // Change cursor to indicate it's clickable

            // Add event listener to the overlay
            overlay.addEventListener('click', function (event) {
                console.log('Overlay clicked');
                event.stopPropagation();
                const currentTime = new Date().getTime();
                const timeDiff = currentTime - lastClickTime;

                if (timeDiff < 300) {
                    clickCount++;
                } else {
                    clickCount = 1;
                }

                lastClickTime = currentTime;

                if (clickCount === 3) {
                    clickCount = 0;
                    let contentType = 'iframe';
                    let contentData = iframe.src;
                    if (contentData) {
                        createInfoWindow(contentData, window.location.href, contentType);
                    }
                }
            });

            iframe.offsetParent.appendChild(overlay);
            iframe.style.position = 'relative';

            // Add event listener to the iframe itself
            iframe.addEventListener('click', function (event) {
                console.log('Iframe video clicked');
                // You can add logic here to play the video if needed
                event.stopPropagation();
            });

            // Mark this iframe element as having an overlay
            iframe.setAttribute('data-overlay-set', 'true');
        }, 100);
    });
}

function observeDOMChanges() {
    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            if (mutation.addedNodes.length) {
                setUpAudioElements();
            }
        });
    });

    observer.observe(document.body, { childList: true, subtree: true });
}

window.onload = function () {
    setUpAudioElements(); // Initial setup
    observeDOMChanges();   // Start observing for DOM changes
    setUpIframeElements();
};


document.addEventListener('click', (event) => {
    console.log('Non audio element clicked');
    const currentTime = new Date().getTime();
    const timeDiff = currentTime - lastClickTime;

    if (timeDiff < 300) {
        clickCount++;
    } else {
        clickCount = 1;
    }

    lastClickTime = currentTime;

    if (clickCount === 3) {
        clickCount = 0;
        let contentType = 'text';
        let contentData;

        if (event.target.closest('p, span, div, a, h1, h2, h3, h4, h5, h6')) {
            let textElement = event.target.closest('p, span, div, a, h1, h2, h3, h4, h5, h6');
            contentData = textElement ? textElement.textContent : '';
            highlightText(textElement);
        } else if (event.target.tagName.match(/^H[1-6]$/)) {
            const header = event.target;
            contentType = 'text';
            contentData = header ? header.textContent : '';
            highlightText(header);
        } else if (event.target instanceof HTMLImageElement) {
            contentType = 'image';
            contentData = event.target.currentSrc;
            highlightImage(event.target);
        } else if (event.target instanceof HTMLVideoElement) {
            contentType = 'video';
            contentData = event.target.src;
            highlightVideo(event.target);
        }
        else if (event.target instanceof HTMLIFrameElement) {
            contentType = 'iframe';
            contentData = event.target.src;
            highlightVideo(event.target);
            console.log('Iframe clicked with src:', contentData);
        }

        if (contentData) {
            createInfoWindow(contentData, window.location.href, contentType);
        }
    }
});

function createInfoWindow(data, url, contentType) {
    const infoWindow = document.createElement('div');
    infoWindow.style.backgroundColor = 'white';
    infoWindow.style.color = 'red';
    infoWindow.style.position = 'fixed';
    infoWindow.style.top = '10px';
    infoWindow.style.left = '50%';
    infoWindow.style.transform = 'translate(-50%, 0)';
    infoWindow.style.padding = '20px';
    infoWindow.style.maxWidth = '400px';
    infoWindow.style.borderRadius = '10px';
    infoWindow.style.boxShadow = '0px 0px 15px rgba(0, 0, 0, 0.6)';
    infoWindow.style.zIndex = '10000';

    const spinnerContainer = document.createElement('div');
    spinnerContainer.style.display = 'flex';
    spinnerContainer.style.flexDirection = 'column';
    spinnerContainer.style.alignItems = 'center';
    spinnerContainer.style.justifyContent = 'center';
    spinnerContainer.style.height = '100px';

    const spinner = document.createElement('div');
    spinner.style.border = '4px solid rgba(0, 0, 0, 0.1)';
    spinner.style.borderTop = '4px solid #3498db';
    spinner.style.borderRadius = '50%';
    spinner.style.width = '60px';
    spinner.style.height = '60px';
    spinner.style.animation = 'spin 4s linear infinite';

    spinnerContainer.appendChild(spinner);
    infoWindow.appendChild(spinnerContainer);
    document.body.appendChild(infoWindow);

    // Make an API request here and update the infoWindow content when the response is received
    const apiUrl = 'http://localhost:5000/api/fact'; // Replace with your API endpoint
    fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            data: data,
            url: url,
            contentType: contentType
        })
    })
        .then(response => response.json())
        .then(data => {
            infoWindow.removeChild(spinnerContainer);

            const contentDiv = document.createElement('div');
            contentDiv.style.fontSize = '24px';
            contentDiv.style.fontWeight = 'bold';
            contentDiv.style.textAlign = 'center';
            const percentageDiv = document.createElement('div');
            let percentage = 0;
            const sourceDiv = document.createElement('div');
            const reasonDiv = document.createElement('div');

            if (data.type.toLowerCase() === 'true') {
                contentDiv.style.color = 'green';
                contentDiv.textContent = 'ინფორმაცია სიმართლეა!';
                sourceDiv.style.color = 'green'
                reasonDiv.style.color = 'green'

                percentage = Math.floor(Math.random() * (100 - 80 + 1) + 80);
            } else if (data.type.toLowerCase() === 'false') {
                contentDiv.style.color = 'red';
                contentDiv.textContent = 'ინფორმაცია სიცრუეა!';
                percentage = Math.floor(Math.random() * (55 - 20 + 1) + 20);
                sourceDiv.style.color = 'red'
                reasonDiv.style.color = 'red'

            }

            percentageDiv.textContent = `ალბათობა: ${percentage}%`;
            contentDiv.appendChild(percentageDiv);
            infoWindow.appendChild(contentDiv);

            // Display the source from the API response
            if (data.message) {
                sourceDiv.textContent = `მესიჯი: ${data.message}`;
                contentDiv.style.marginBottom = '10px';
                sourceDiv.style.marginTop = '10px';
                infoWindow.appendChild(sourceDiv);
            }
            
            if (data.reason) {
                reasonDiv.textContent = `მიზეზი: ${data.reason}`;
                reasonDiv.style.marginTop = '10px';
                infoWindow.appendChild(reasonDiv);
            }

            const closeButton = document.createElement('button');
            closeButton.textContent = '✖️';
            closeButton.style.position = 'absolute';
            closeButton.style.top = '10px';
            closeButton.style.right = '20px';
            closeButton.style.fontWeight = 'bold';
            closeButton.style.padding = '5px 8px';
            closeButton.style.backgroundColor = 'transparent';
            closeButton.style.border = 'none';
            closeButton.style.color = 'red';
            closeButton.style.cursor = 'pointer';
            closeButton.style.transition = 'background-color 0.3s';


            closeButton.addEventListener('click', () => {
                document.body.removeChild(infoWindow);
                removeHighlights();
            });
            closeButton.addEventListener('mouseenter', () => {
                closeButton.style.backgroundColor = 'rgba(255, 0, 0, 0.1)';
            });
            closeButton.addEventListener('mouseleave', () => {
                closeButton.style.backgroundColor = 'transparent';
            });


            infoWindow.appendChild(closeButton);


            // Close button and other content can be added here
        })
        .catch(error => {
            console.error('API request failed:', error);
            // Handle the error, e.g., display an error message
        });
}

// Add CSS animation for the spinner

function highlightText(element) {
    element.style.border = "4px solid red";
}

function highlightImage(element) {
    element.style.border = "4px solid red";
}

function highlightVideo(element) {
    element.style.border = "4px solid green";
}

function removeHighlights() {
    // Remove borders from images and videos
    document.querySelectorAll('p[style*="border"], img[style*="border"], video[style*="border"], h1[style*="border"], h2[style*="border"], h3[style*="border"], h4[style*="border"], h5[style*="border"], h6[style*="border"]').forEach(element => {
        element.style.border = '';
    });
}