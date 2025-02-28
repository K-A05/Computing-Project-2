function appendMessage(sender, message) {
    let chatbox = document.getElementById("chatbox");
    let messageElement = document.createElement("p");
    messageElement.textContent = sender + ": " + message;
    chatbox.appendChild(messageElement);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function getBotResponse(input) {
    let responses = {
        "hello": "Hi there! How can I help you?",
        "how are you": "I'm just a bot, but I'm doing great!",
        "bye": "Goodbye! Have a great day!"
    };
    return responses[input.toLowerCase()] || "I'm not sure how to respond to that.";
}

function handleKeyPress(event) {
    if (event.key === "Enter") {
        let userInput = document.getElementById("userInput").value;
        if (userInput.trim() !== "") {
            appendMessage("You", userInput);
            let botResponse = getBotResponse(userInput);
            setTimeout(() => appendMessage("Bot", botResponse), 500);
            document.getElementById("userInput").value = "";
        }
    }
}
