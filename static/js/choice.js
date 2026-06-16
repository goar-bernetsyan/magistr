// static/js/choice.js

document.addEventListener('DOMContentLoaded', () => {

    // --- КОД ДЛЯ КНОПКИ "ДАТАСЕТ" ---
    const datasetButton = document.querySelector('.dataset-btn');
    const datasetModal = document.getElementById('datasetViewerModal');

    if (datasetButton && datasetModal) {
        datasetButton.addEventListener('click', () => {
            datasetModal.style.display = 'flex';

            // Получаем имя файла из заголовка (он уже подставлен Flask'ом)
            const filenameSpan = document.querySelector('.modal-header h3');
            let datasetFilename = '{{ filename }}'; // По умолчанию

            // Если заголовок уже был изменен (например, при повторном открытии)
            if (filenameSpan && filenameSpan.textContent.includes('Просмотр датасета: ')) {
                const match = filenameSpan.textContent.match(/Просмотр датасета: (.+)/);
                if (match) {
                    datasetFilename = match[1];
                }
            }

            if (datasetFilename !== 'None' && datasetFilename !== '') {
                loadDatasetPreview(datasetFilename);
            } else {
                // Если имя файла не передалось, показываем сообщение
                const tableBody = document.getElementById('datasetTable');
                const loadingMessage = document.querySelector('.modal-body p');
                if (loadingMessage) loadingMessage.style.display = 'none';
                tableBody.innerHTML = `<tr><td colspan="100%" style="text-align:center; color:red;">Имя файла датасета не определено.</td></tr>`;
            }
        });
    }

    // Закрытие модального окна по клику вне него или на крестик
    // Предполагается, что closeDatasetViewer() также есть в choice.js или scripts.js
    // Если нет, то этот код будет работать:
    window.addEventListener('click', function(event) {
        if (event.target === datasetModal) {
            datasetModal.style.display = 'none';
        }
    });

    const closeButtons = document.querySelectorAll('.close-button');
    closeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            datasetModal.style.display = "none";
        });
    });

    // --- КОД ДЛЯ РАСКРЫВАЮЩЕГОСЯ СПИСКА ---
    const actionItems = document.querySelectorAll('.action-item.has-submenu');

    actionItems.forEach(item => {
        item.addEventListener('click', function(e) {
            // Проверяем, что клик был по самому пункту, а не по вложенному чекбоксу
            if (e.target.tagName !== 'INPUT') {
                this.classList.toggle('open');
                const submenu = this.querySelector('.submenu');
                // При клике на главный пункт, он раскрывает/скрывает подменю
                // Если клик на чекбокс, то подменю не должно меняться
                if (submenu) { // Убедимся, что submenu существует
                     submenu.style.display = submenu.style.display === 'block' ? 'none' : 'block';
                }
            }
        });
    });
});

// Функция отправки выбранных действий на сервер
function submitActions() {
  const selectedActions = [];
  document.querySelectorAll('input[name="action"]:checked').forEach(cb => {
     selectedActions.push(cb.value);
  });

  if (selectedActions.length === 0) {
     alert("Пожалуйста, выберите хотя бы одно действие.");
     return;
  }

   // === НОВАЯ СТРОКА ДЛЯ ОТЛАДКИ ===
  console.log("Отправляемые данные:", {
     filename: "{{ filename }}",
     actions: selectedActions
  });

  fetch('/start-process', { // <-- ИЗМЕНЕННЫЙ URL
     method: 'POST',
     headers: {'Content-Type': 'application/json'},
     body: JSON.stringify({
         filename: uploadedFilename,
         actions: selectedActions
     })
  })
  .then(response => {
      // Проверяем, был ли HTTP-статус ответа успешным (200-299)
      if (response.ok) {
          // Если да - просто переходим на страницу результата
          window.location.href = '/result';
      } else {
          // Если нет (например, ошибка 400 или 500) - показываем ошибку
          return response.json().then(errData => {
              throw new Error(errData.message || 'Ошибка сервера при старте процесса');
          });
      }
  })
  .catch(error => {
      // Этот блок выполнится, если была сетевая ошибка или мы сами вызвали throw выше
      console.error("Ошибка при отправке данных:", error);
      alert("Ошибка старта процесса: " + error.message);
  });
}

// --- ФУНКЦИЯ ЗАГРУЗКИ ПРЕВЬЮ ДАТАСЕТА ---
function loadDatasetPreview(filename) {
    const tableBody = document.getElementById('datasetTable');
    const loadingMessage = document.querySelector('.modal-body p'); // Параграф "Загрузка данных..."

    // Показываем сообщение о загрузке и очищаем таблицу
    if (loadingMessage) loadingMessage.style.display = 'block';
    tableBody.innerHTML = '';

    // Отправляем запрос на сервер (используем POST, как в вашем app.py)
    fetch('/view-dataset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: filename }), // Передаем имя файла в теле запроса
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.message || 'Ошибка сервера при получении данных') });
        }
        return response.json();
    })
    .then(data => {
        // Скрываем сообщение о загрузке
        if (loadingMessage) loadingMessage.style.display = 'none';

        if (data.status === 'success') {
            // Если данные получены успешно, строим таблицу
            const headers = data.headers;
            const preview = data.preview;

            let tableHTML = '<thead><tr>';
            headers.forEach(header => {
                tableHTML += `<th>${header}</th>`;
            });
            tableHTML += '</tr></thead><tbody>';

            preview.forEach(row => {
                tableHTML += '<tr>';
                row.forEach(cell => {
                    tableHTML += `<td>${cell}</td>`;
                });
                tableHTML += '</tr>';
            });
            tableHTML += '</tbody>';

            tableBody.innerHTML = tableHTML;

            // Обновляем заголовок модального окна с именем файла
            const modalTitle = document.querySelector('.modal-header h3');
            if (modalTitle) {
                modalTitle.textContent = `Просмотр датасета: ${filename}`;
            }

        } else if (data.status === 'error') {
            // Если сервер вернул ошибку
            tableBody.innerHTML = `<tr><td colspan="100%" style="text-align:center; color:red;">Ошибка: ${data.message}</td></tr>`;
            const modalTitle = document.querySelector('.modal-header h3');
            if (modalTitle) {
                modalTitle.textContent = `Просмотр датасета: Ошибка`;
            }
        }
    })
    .catch(error => {
        console.error("Ошибка сети или при получении данных:", error);
        if (loadingMessage) loadingMessage.style.display = 'none';
        tableBody.innerHTML = `<tr><td colspan="100%" style="text-align:center; color:red;">Ошибка сети. Проверьте консоль. Детали: ${error.message}</td></tr>`;
        const modalTitle = document.querySelector('.modal-header h3');
        if (modalTitle) {
            modalTitle.textContent = `Просмотр датасета: Ошибка`;
        }
    });
}

// Вспомогательная функция для закрытия модального окна (если она не определена глобально)
function closeDatasetViewer() {
    const datasetModal = document.getElementById('datasetViewerModal');
    if (datasetModal) {
        datasetModal.style.display = 'none';
    }
}