let currentPlant = null;
let conversationContext = [];
let sessionId = generateSessionId(); // Generate a unique session ID

function generateSessionId() {
    return 'session_' + Math.random().toString(36).substring(2, 15);
}

window.onload = function() {
    conversationContext.push({
        role: "model",
        parts: [{ text: "Hello! I can help you identify plants and answer questions about them. Start by uploading a clear image of a plant you want to identify." }]
    });
    
    marked.setOptions({
        breaks: true,
        gfm: true,
        sanitize: false,
        mangle: false,
        headerIds: false,
    });
};

function sanitizeHTML(html) {
    const temp = document.createElement('div');
    temp.innerHTML = html;
    
    const scripts = temp.querySelectorAll('script, iframe, object, embed');
    scripts.forEach(script => script.remove());
    
    const allElements = temp.querySelectorAll('*');
    allElements.forEach(element => {
        const attributes = element.attributes;
        for (let i = attributes.length - 1; i >= 0; i--) {
            const attributeName = attributes[i].name;
            if (attributeName.startsWith('on') || attributeName === 'href' && attributes[i].value.startsWith('javascript:')) {
                element.removeAttribute(attributeName);
            }
        }
    });
    
    return temp.innerHTML;
}

async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    addMessage(message, 'user');
    
    // Clear input field
    messageInput.value = '';
    
    // Loading element
    const loadingElement = document.createElement('div');
    loadingElement.className = 'loading';
    loadingElement.textContent = 'Thinking...';
    chatContainer.appendChild(loadingElement);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId,
                current_plant: currentPlant
            })
        });
        
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove loading message
        chatContainer.removeChild(loadingElement);
        
        if (data.response) {
            addMarkdownMessage(data.response, 'bot');
            
            if (data.conversation_context) {
                conversationContext = data.conversation_context;
            }
        } else if (data.error) {
            addMessage(`Error: ${data.error}. Please try again.`, 'bot');
        } else {
            addMessage("I'm sorry, I couldn't generate a response. Please try again.", 'bot');
        }
        
    } catch (error) {
        console.error('Error calling API:', error);
        chatContainer.removeChild(loadingElement);
        addMessage(`There was an error processing your request: ${error.message}. Please try again.`, 'bot');
    }
}

function addMessage(message, sender) {
    const chatContainer = document.getElementById('chatContainer');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    
    if (sender === 'user') {
        messageElement.classList.add('user-message');
        messageElement.textContent = message;
    } else {
        messageElement.classList.add('bot-message');
        messageElement.innerHTML = message;
    }
    
    chatContainer.appendChild(messageElement);
    
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addMarkdownMessage(markdownText, sender) {
    const chatContainer = document.getElementById('chatContainer');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    
    if (sender === 'user') {
        messageElement.classList.add('user-message');
        messageElement.textContent = markdownText; 
    } else {
        messageElement.classList.add('bot-message');
        const htmlContent = marked.parse(markdownText);
        messageElement.innerHTML = sanitizeHTML(htmlContent);
    }
    
    chatContainer.appendChild(messageElement);
    
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function identifyPlant() {
    const imageUpload = document.getElementById('imageUpload');

    // imageUpload.click();
    // const previewImage = document.getElementById('previewImage');
    // const latitude = document.getElementById('latitudeInput').value.trim();
    // const longitude = document.getElementById('longitudeInput').value.trim();
    // const similarImages = document.getElementById('similarImagesToggle').checked;
    
    if (imageUpload.files && imageUpload.files[0]) {
        const reader = new FileReader();
        
        reader.onload = async function(e) {
            // previewImage.src = e.target.result;
            // previewImage.style.display = 'block';
            
            addMessage("I've uploaded a plant image for identification", 'user');
            
            addMessage("Analyzing your plant image... Please wait.", 'bot');
            
            const base64Image = e.target.result.split(',')[1];
        
            const loadingElement = document.createElement('div');
            loadingElement.className = 'loading';
            loadingElement.textContent = 'Identifying plant...';
            chatContainer.appendChild(loadingElement);
            
            try {
                const response = await fetch('/api/identify-plant', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        image: base64Image,
                        session_id: sessionId,
                        // latitude: latitude,
                        // longitude: longitude,
                        // similar_images: similarImages
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`API request failed: ${response.status}`);
                }
                
                const data = await response.json();
                console.log('Plant Identification API Response:', data);
                
                chatContainer.removeChild(loadingElement);
                
                if (data.conversation_context) {
                    conversationContext = data.conversation_context;
                }
                
                if (data.plant_data && data.plant_data.result && 
                    data.plant_data.result.classification && 
                    data.plant_data.result.classification.suggestions && 
                    data.plant_data.result.classification.suggestions.length > 0) {
                    
                    const topResult = data.plant_data.result.classification.suggestions[0];
                    
                    currentPlant = {
                        name: topResult.name,
                        probability: topResult.probability,
                        commonNames: topResult.details && topResult.details.common_names ? topResult.details.common_names : [],
                        description: topResult.details && topResult.details.description ? topResult.details.description.value : null,
                        watering: topResult.details && topResult.details.watering ? topResult.details.watering.value : null
                    };
                    
                    let plantInfo = `<strong>Plant Identified:</strong> ${topResult.name}<br>`;
                    
                    if (topResult.probability) {
                        plantInfo += `<strong>Confidence:</strong> ${(topResult.probability * 100).toFixed(1)}%<br>`;
                    }
                    
                    if (topResult.details && topResult.details.common_names && topResult.details.common_names.length > 0) {
                        plantInfo += `<strong>Common Names:</strong> ${topResult.details.common_names.join(', ')}<br>`;
                    }
                    
                    if (topResult.details && topResult.details.description) {
                        plantInfo += `<strong>Description:</strong> ${topResult.details.description.value}<br>`;
                    }
                    
                    if (topResult.details && topResult.details.watering) {
                        plantInfo += `<strong>Watering Needs:</strong> ${topResult.details.watering.value}<br>`;
                    }
                    
                    if (data.plant_data.result.classification.suggestions.length > 1) {
                        plantInfo += `<br><strong>Other Possibilities:</strong><br>`;
                        for (let i = 1; i < Math.min(3, data.plant_data.result.classification.suggestions.length); i++) {
                            const suggestion = data.plant_data.result.classification.suggestions[i];
                            plantInfo += `- ${suggestion.name} (${(suggestion.probability * 100).toFixed(1)}%)<br>`;
                        }
                    }
                    
                    // if (similarImages && data.plant_data.result.similar_images && data.plant_data.result.similar_images.length > 0) {
                    //     plantInfo += `<br><strong>Similar Images:</strong><br>`;
                    //     for (let i = 0; i < Math.min(2, data.plant_data.result.similar_images.length); i++) {
                    //         const similarImage = data.plant_data.result.similar_images[i];
                    //         if (similarImage.url) {
                    //             plantInfo += `<a href="${similarImage.url}" target="_blank">View similar plant ${i+1}</a><br>`;
                    //         }
                    //     }
                    // }
                    
                    plantInfo += `<br>You can now ask me questions about this ${topResult.name} plant!`;
                    
                    addMessage(plantInfo, 'bot');
                    
                    if (data.welcome_message) {
                        addMarkdownMessage(data.welcome_message, 'bot');
                    }
                    
                } else if (data.error) {
                    addMessage(`Error: ${data.error}. Please try with a clearer image or a different plant.`, 'bot');
                } else {
                    addMessage("I couldn't identify this plant from the image. Please try with a clearer image or a different plant.", 'bot');
                }
                
            } catch (error) {
                console.error('Error:', error);
                chatContainer.removeChild(loadingElement);
                addMessage(`There was an error identifying the plant: ${error.message}. Please try again.`, 'bot');
            }
        };
        
        reader.readAsDataURL(imageUpload.files[0]); 
    } else {
        addMessage("Please select an image first.", 'bot');
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}