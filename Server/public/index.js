window.addEventListener('DOMContentLoaded', (event) => {
    // document.querySelectorAll('.user-name').forEach((elem) => {
    //     elem.addEventListener('click', (event) => {
    //         alert(event.target.innerHTML);
    //     });
    // });

    document.querySelector('.send-button1').addEventListener('click', (event) => {
        const newElement = document.createElement('li');
        const text = document.querySelector('.input-text').value;
        newElement.innerHTML = text;
        document.querySelector('.user-list').appendChild(newElement);
        fetch('/api/user-add',{ method: 'POST',headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: text }) });
        window.location.reload();
    });

    document.querySelector('.send-button2').addEventListener('click', (event) => {
        const text = document.querySelector('.delete-text').value;
        let list_element = document.getElementById(text);
        list_element.remove();
        fetch('/api/user-delete',{ method: 'POST',headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: text }) });
        window.location.reload();
    });

    document.querySelector('.send-button3').addEventListener('click', (event) => {
        const pre = document.querySelector('.previous-text').value;
        const next = document.querySelector('.next-text').value;
        const list_element = document.getElementById(pre);
        list_element.innerHTML = next;
        list_element.id = next;
        fetch('/api/user-update',{ method: 'POST',headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: pre, new_name: next}) });
        window.location.reload();
    });
});