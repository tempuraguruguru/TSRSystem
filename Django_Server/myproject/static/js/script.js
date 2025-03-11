// window.addEventListener("beforeunload", (event) => {
//     // サーバーにリクエストを送信
//     navigator.sendBeacon("/close-tab/", JSON.stringify({ action: "tab_closed" }));
// });


window.addEventListener("unload", function (event) {
    // タブが閉じられる直前にPOSTリクエストを送信する
    fetch('/close_tab/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': '{{ csrf_token }}'
        },
        body: JSON.stringify({ "action": "tab_closed" })
    });
});