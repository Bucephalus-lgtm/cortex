document.getElementById('ask-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const inputField = document.getElementById('query-input');
    const query = inputField.value.trim();
    if (!query) return;

    // Add User Message
    addMessage(query, 'user');
    inputField.value = '';

    // Add Loading State
    const loadingId = addTypingIndicator();

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: query })
        });
        
        const data = await response.json();
        removeElement(loadingId);

        if (response.ok) {
            addMessage(data.answer, 'ai', data.sources);
        } else {
            addMessage('⚠️ Something went wrong connecting to the backend.', 'ai');
        }

    } catch (err) {
        removeElement(loadingId);
        addMessage('⚠️ Network Error: Unable to reach the Cortex API.', 'ai');
        console.error(err);
    }
});

function addMessage(text, role, sources = []) {
    const chatBox = document.getElementById('chat-box');
    
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message');
    msgDiv.classList.add(role === 'user' ? 'user-message' : 'ai-message');

    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    avatar.textContent = role === 'user' ? '🧑‍💻' : '🧠';

    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    
    // Simple markdown hack logic to bold text headers
    let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    bubble.innerHTML = formattedText;

    if (sources && sources.length > 0) {
        const sourceBox = document.createElement('div');
        sourceBox.classList.add('sources-box');
        
        sources.forEach(src => {
            const badge = document.createElement('span');
            badge.classList.add('source-badge');
            // Remove full path noise just show filename
            badge.textContent = src.split('/').pop().split('\\').pop(); 
            sourceBox.appendChild(badge);
        });

        bubble.appendChild(sourceBox);
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addTypingIndicator() {
    const id = 'typing-' + Date.now();
    const chatBox = document.getElementById('chat-box');
    
    const msgDiv = document.createElement('div');
    msgDiv.id = id;
    msgDiv.classList.add('message', 'ai-message');

    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    avatar.textContent = '🧠';

    const bubble = document.createElement('div');
    bubble.classList.add('bubble', 'typing-indicator');
    bubble.style.display = 'flex';
    
    bubble.innerHTML = `<div class="dot"></div><div class="dot"></div><div class="dot"></div>`;

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatBox.appendChild(msgDiv);
    
    chatBox.scrollTop = chatBox.scrollHeight;
    return id;
}

function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}
