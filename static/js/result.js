// static/js/result.js

document.addEventListener('DOMContentLoaded', function () {
    const resultsContainer = document.getElementById('results-container');
    const exportBtn = document.getElementById('export-btn');




    // --- НОВОЕ: Добавим CSS для блоков результатов прямо в JS для наглядности ---
    const style = document.createElement('style');
    style.textContent = `
        .result-block {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 20px;
            padding: 15px;
            background-color: #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        /* НОВОЕ ПРАВИЛО: Делаем контейнер гибким */
        #results-container {
            display: flex; /* Включаем Flexbox */
            flex-direction: column; /* Задаем направление сверху вниз */
            gap: 20px; /* Добавляем отступы между блоками (современная альтернатива margin-bottom) */
        }
        .result-block h2 {
            margin-top: 0;
            color: #333;
        }
        .loading-spinner {
            text-align: center;
            font-size: 1.2em;
            padding: 20px;
        }
        .error-message {
            color: #d32f2f;
            background-color: #ffebee;
            padding: 15px;
            border-radius: 4px;
        }
    `;
    document.head.appendChild(style);




    // Удаляем начальный текст "Ожидание начала обработки..."
    const initialText = resultsContainer.querySelector('.loading-spinner');
    if (initialText) {
        initialText.remove();
    }



    // Функция для загрузки следующего шага (рекурсивная)
    function loadNextStep() {
        fetch('/get-next-action', { method: 'GET' })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'finished') {
                    // Все шаги выполнены
                    const block = document.createElement('div');
                    block.className = 'result-block';
                    block.innerHTML = `
                        <h2>Обработка завершена!</h2>
                        <p>Все выбранные действия выполнены. Вы можете экспортировать результаты.</p>
                    `;
                    resultsContainer.appendChild(block);
                    exportBtn.style.display = 'inline-block'; // Показываем кнопку экспорта
                    return;
                }

                if (data.status === 'need_params') {
                    showParamsForm(data);
                } else if (data.status === 'execute_now') {
                    // --- СОЗДАЕМ НОВЫЙ БЛОК СО СПИННЕРОМ ---
                    const block = document.createElement('div');
                    block.className = 'result-block';
                    block.innerHTML = '<div class="loading-spinner">Выполняется анализ...</div>';

                    // Добавляем его в конец страницы
                    resultsContainer.appendChild(block);

                    // Прокручиваем к новому блоку
                    block.scrollIntoView({ behavior: 'smooth' });

                    // Запускаем действие.
                    // Оно само найдет этот блок и заменит в нем спиннер на результат.
                    executeAction(data.action, {}, loadNextStep);
                } else {
                    console.error('Неизвестный статус:', data);
                }
            })
            .catch(error => {
                console.error('Ошибка:', error);
                const block = document.createElement('div');
                block.className = 'result-block error-message';
                block.textContent = 'Произошла ошибка при загрузке шага. Проверьте консоль.';
                resultsContainer.appendChild(block);
            });
    }

    // Функция для отображения формы параметров
    function showParamsForm(data) {
        let formHTML = `
            <div class="result-block">
                <h5>Параметры для действия: ${data.action}</h5>
                <form id="action-form" class="params-form">
                    <div class="params-fields">
        `;


        const action = data.action;
        const schema = data.params_schema;


        if (schema.type === 'checkbox') {
            formHTML += `<p>${schema.label}</p>`;
            schema.options.forEach(option => {
                formHTML += `
                    <label>
                        <input type="checkbox" name="${schema.name}" value="${option}">
                        ${option}
                    </label><br>
                `;
            });
        }

        else if (schema.type === 'radio') {
            formHTML += `<p>${schema.label}</p>`;
            schema.options.forEach(option => {
                // Добавлено: required (атрибут обязательности)
                formHTML += `
                    <label>
                        <input type="radio" name="${schema.name}" value="${option.value}" required>
                        ${option.label}
                    </label><br>
                `;
            });
        }

        else if (action === 'data_cleaning') {
            // Формируем HTML для сложной формы с выбором метода очистки для каждого столбца
            formHTML += `<p>${schema.label}</p>`;

            const columnsList = data.params_schema.columns_list;

            // УБИРАЕМ "--Не обрабатывать--". "пропустить" теперь по умолчанию (пустое значение select)
            const actionsList = [
                "заменить на 0",
                "заменить на среднее",
                "заменить на медиану",
                "заменить на самое частое значение",
                "предсказать значение с помощью KNNImputer",
                "заполнить предыдущим значением",
                "заполнить следующим значением"
            ];

            formHTML += '<table class="cleaning-table"><thead><tr><th>Столбец</th><th>Метод очистки</th></tr></thead><tbody>';

            columnsList.forEach(col => {
                formHTML += `<tr>
                                <td>${col}</td>
                                <td>
                                    <select name="cleaning_${col}">
                                        <!-- ПУСТОЕ ЗНАЧЕНИЕ ПО УМОЛЧАНИЮ ("пропустить") -->
                                        <option value="" selected>пропустить</option>`;

                actionsList.forEach(action => {
                    formHTML += `<option value="${action}">${action}</option>`;
                });

                formHTML += `      </select>
                                </td>
                             </tr>`;
            });

            formHTML += `</tbody></table>`;
        }

        else if (action === 'unique_value_analysis') {
            const columnsList = data.params_schema.columns_list;

            formHTML += `<p>${schema.label}</p>`;

            // Радиокнопки для выбора типа частот (Абсолютные / Относительные)
            formHTML += `
                <div>
                    <p>Тип частот:</p>
                    <label>
                        <input type="radio" name="normalize" value="false" checked>
                        Абсолютные (количество)
                    </label><br>
                    <label>
                        <input type="radio" name="normalize" value="true">
                        Относительные (доли, %)
                    </label>
                </div>
                <hr>
                <p>Выберите столбцы для анализа частот:</p>
                <div class="checkbox-group">`;

            columnsList.forEach(col => {
                formHTML += `
                    <label>
                        <input type="checkbox" name="columns" value="${col}">
                        ${col}
                    </label><br>`;
            });

            formHTML += `
                </div>
            `;
        }

        else if (action === 'pivot_tables') {
            const columnsList = data.params_schema.columns_list;

            formHTML += `<p>${data.params_schema.label}</p>`;

            // 1. Выбор режима: Сводная таблица или Группировка
            formHTML += `
                <div>
                    <p>Режим анализа:</p>
                    <label>
                        <input type="radio" name="mode" value="pivot" checked>
                        Сводная таблица (Pivot Table)
                    </label><br>
                    <label>
                        <input type="radio" name="mode" value="group">
                        Группировка (Group By)
                    </label>
                </div>
                <hr>
            `;

            // 2. Выбор элементов для списков (используем мультиселекты)
            formHTML += `
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">

                    <div>
                        <label>Строки (индекс/группировка): <span style="color: red;">*</span></label>
                        <select name="row_categories" multiple required style="width: 100%; height: 100px;">
                            ${columnsList.map(col => `<option value="${col}">${col}</option>`).join('')}
                        </select>
                    </div>

                    <div>
                        <label>Столбцы (для Pivot Table):</label>
                        <select name="col_categories" multiple style="width: 100%; height: 100px;">
                            ${columnsList.map(col => `<option value="${col}">${col}</option>`).join('')}
                        </select>
                    </div>

                    <div>
                        <label>Значения (для анализа): <span style="color: red;">*</span></label>
                        <select name="columns" multiple required style="width: 100%; height: 100px;">
                            ${columnsList.map(col => `<option value="${col}">${col}</option>`).join('')}
                        </select>

                        <label>Функции агрегации:</label>
                        <select name="functions" multiple style="width: 100%; height: 50px;">
                            <option value="mean">Среднее (mean)</option>
                            <option value="sum">Сумма (sum)</option>
                            <option value="count">Количество (count)</option>
                            <option value="min">Минимум (min)</option>
                            <option value="max">Максимум (max)</option>
                            <option value="std">Стандартное отклонение (std)</option>
                        </select>
                    </div>

                </div>
            `;
        }

        else if (schema.type === 'complex_ml') {
            const columnsList = data.params_schema.columns_list;
            const modelDisplayName = data.params_schema.model_name
                .replace('R_', '')
                .replace('C_', '')
                .replace(/_/g, ' ') // Заменяет все подчеркивания на пробелы
                .replace(/\b\w/g, l => l.toUpperCase()); // Делает первую букву каждого слова заглавной

            formHTML += `
                <div>
                    <h5>Модель: ${modelDisplayName}</h5>
                    <p>${data.params_schema.label}</p>

                    <label for="target_col_select">Выберите целевой столбец (что предсказываем):</label>
                    <select name="target_col" id="target_col_select" required>
                        <option value="" disabled selected>Выберите столбец...</option>
                        ${columnsList.map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
            `;
        }




        formHTML += `
                    </div>
                    <button type="submit" class="ready-btn">Выполнить</button>
                </form>
            </div>
        `;

        const block = document.createElement('div');
        block.className = 'result-block';
        block.innerHTML = formHTML;

        resultsContainer.appendChild(block);
        block.scrollIntoView({ behavior: 'smooth' });

        // Обработчик отправки формы
        const form = block.querySelector('#action-form');
        form.addEventListener('submit', function (e) {
            e.preventDefault();

            const formData = new FormData(form);
            const params = {};

             if (action === 'data_cleaning') {
                 for (let [key, value] of formData.entries()) {
                     if (key.startsWith('cleaning_')) {
                         params[key] = value;
                     }
                 }
             } else if (action === 'unique_value_analysis') {
                 const normalizeValue = formData.get('normalize');
                 if (normalizeValue !== null) params['normalize'] = normalizeValue;
                 params['columns'] = Array.from(formData.getAll('columns'));
             } else if (action === 'pivot_tables') {
                 params['mode'] = formData.get('mode');
                 params['row_categories'] = Array.from(formData.getAll('row_categories'));
                 params['col_categories'] = Array.from(formData.getAll('col_categories'));
                 params['columns'] = Array.from(formData.getAll('columns'));
                 params['functions'] = Array.from(formData.getAll('functions'));
             } else if (action.startsWith('C_') || action.startsWith('R_')) {
                 // Просто берем значение из выпадающего списка target_col
                 const targetColValue = formData.get('target_col');
                 if (targetColValue) { // Проверяем, что что-то выбрано
                    params['target_col'] = targetColValue;
                 }
             } else if (schema.type === 'checkbox' || schema.type === 'radio') {
                 // было: params[schema.name] = Array.from(formData.getAll(schema.name));
                 //стало:
                 const values = Array.from(formData.getAll(schema.name));
                 params[schema.name] = schema.type === 'radio' ? values[0] : values; //Для radio берем первое значение, для checkbox оставляем массив
             }

             executeAction(action, params, loadNextStep);

             const button = form.querySelector('button');
             if (button) {
                 button.disabled = true;
                 button.textContent = 'Выполняется...';
             }
        });
    }

    // Функция для выполнения действия и показа результата
    function executeAction(actionName, params, callback) {
        // --- ДОБАВЛЕНА ПРОВЕРКА ФОРМЫ ---
        const form = document.getElementById('action-form');
        // Проверяем, существует ли форма. Если нет — значит, параметры не нужны.
        if (form) {
            // Если форма существует, проверяем её валидность
            if (!form.checkValidity()) {
                // Если форма невалидна (например, радиобаттон не выбран), подсвечиваем ошибку
                form.reportValidity();
                return; // Останавливаем выполнение
            }
        }
        // --- КОНЕЦ ПРОВЕРКИ ---



        fetch('/submit-action-params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: actionName, params: params })
        })
            .then(response => response.json())
            .then(data => {
                 // Находим ПОСЛЕДНИЙ блок на странице. Это тот самый блок со спиннером,
                 // который мы только что создали в loadNextStep.
                 const lastBlock = resultsContainer.lastElementChild;

                 if (data.status === 'success') {
                     // ЗАМЕНЯЕМ содержимое блока. Спиннер исчезает, появляется результат.
                     lastBlock.innerHTML = data.result_html;

                     lastBlock.scrollIntoView({ behavior: 'smooth' });

                     setTimeout(callback, 500);
                 } else {
                     lastBlock.innerHTML = `<p class="error-message">Ошибка: ${data.message}</p>`;
                 }
            })
            .catch(error => {
                 console.error('Ошибка выполнения действия:', error);
                 const lastBlock = resultsContainer.lastElementChild;
                 if(lastBlock) {
                     lastBlock.innerHTML = `<p class="error-message">Ошибка сети при выполнении действия.</p>`;
                 }
            });
    }

    // Кнопка экспорта в PDF (html2pdf.js)
    exportBtn.addEventListener('click', function () {
        const element = document.getElementById('results-container');

        const opt = {
            margin: 0.5,
            filename: 'results.pdf',
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2 },
            jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
        };

        html2pdf().set(opt).from(element).save();
    });

    // Стартуем процесс при загрузке страницы result.html
    loadNextStep();
});
