// static/js/home.js

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.querySelector('.upload-btn'); // Получаем кнопку


    if (fileInput && uploadButton) {
        // Обработчик клика по кнопке "ЗАГРУЗИТЬ"
        // Убираем inline onclick из HTML, переносим сюда
        uploadButton.addEventListener('click', () => {
            fileInput.click();
        });

        // Обработчик выбора файла
        fileInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                console.log('Выбран файл:', file.name);

                const formData = new FormData();
                formData.append('dataset_file', file); // Добавляем файл в FormData

                // Отправка файла на сервер через AJAX
                fetch('/upload-ajax', { // URL вашего AJAX-обработчика
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json()) // Ожидаем JSON ответ
                .then(data => {
                    if (data.status === 'success') {
                        console.log('Сервер ответил:', data.message);
                        // Перенаправление на страницу choice
                        if (data.redirect_url) {
                            window.location.href = data.redirect_url;
                        } else {
                            alert('Файл "' + file.name + '" успешно загружен! Перенаправление...');
                        }
                    } else {
                        console.error('Ошибка при загрузке файла:', data.message);
                        alert('Произошла ошибка при загрузке файла: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Ошибка сети при загрузке файла:', error);
                    alert('Произошла ошибка сети при загрузке файла.');
                });
            }
        });
    }
});