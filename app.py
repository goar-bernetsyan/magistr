from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import os
from werkzeug.utils import secure_filename
import pandas as pd

import matplotlib
matplotlib.use('Agg')
# Теперь можно импортировать остальные библиотеки
import matplotlib.pyplot as plt

import seaborn as sns
import io
import sys
import base64
import numpy as np
#import openpyxl # для экселя был, в итоге ошибка исчезла, файл стал отображаться, но некоррктно

import joblib
import uuid

from flask import send_file


# try:
#     plt.figure(figsize=(15, 15), dpi=90)
#     # ... ваш код для построения графика ...
#     plt.savefig('путь_к_файлу.png')
#     plt.close() # Обязательно закрываем фигуру
# except Exception as e:
#     # Логируем ошибку или обрабатываем её
#     print(f"Ошибка при построении графика: {e}")




import inspect


# импорты для предсказания NaN
from sklearn.impute import KNNImputer

# импорты для блока МО
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV # либо вместо GridSearchCV - RandomizedSearchCV

from sklearn.linear_model import LinearRegression as lm
from sklearn.linear_model import LogisticRegression as lr

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, confusion_matrix

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

sepp = ';' # разделитель столбцов в файле
# Словарь для сопоставления имени модели с классом sklearn/xgboost/catboost

MODEL_MAP = {
    'R_linear_regression': {
        'model': lm,
        'name': 'Линейная регрессия'
    },
    'R_decision_tree': {
        'model': DecisionTreeRegressor,
        'name': 'Деревья решений (регрессия)'
    },
    'R_random_forest': {
        'model': RandomForestRegressor,
        'name': 'Случайный лес (регрессия)'
    },
    'R_gradient_boosting_xg': {
        'model': XGBRegressor,
        'name': 'Градиентный бустинг XGBoost (регрессия)'
    },
    'R_gradient_boosting_cat': {
        'model': CatBoostRegressor,
        'name': 'Градиентный бустинг CatBoost (регрессия)'
    },
    'R_svm': {
        'model': SVR,
        'name': 'Метод опорных векторов (SVM) (регрессия)'
    },
    'R_knn': {
        'model': KNeighborsRegressor,
        'name': 'Метод k-ближайших соседей (KNN) (регрессия)'
    },

    'C_logistic_regression': {
        'model': lr,
        'name': 'Логистическая регрессия'
    },
    'C_decision_tree': {
        'model': DecisionTreeClassifier,
        'name': 'Деревья решений (классификация)'
    },
    'C_random_forest': {
        'model': RandomForestClassifier,
        'name': 'Случайный лес (классификация)'
    },
    'C_gradient_boosting_xg': {
        'model': XGBClassifier,
        'name': 'Градиентный бустинг XGBoost (классификация)'
    },
    'C_gradient_boosting_cat': {
        'model': CatBoostClassifier,
        'name': 'Градиентный бустинг CatBoost (классификация)'
    },
    'C_svm': {
        'model': SVC,
        'name': 'Метод опорных векторов (SVM) (классификация)'
    },
    'C_knn': {
        'model': KNeighborsClassifier,
        'name': 'Метод k-ближайших соседей (KNN) (классификация)'
    },
}


param_grids = {
    'R_linear_regression': {
        'fit_intercept': [True, False]
    },
    'R_decision_tree': {
        'max_depth': [3, 5, 7, None],  # уменьшили верхний предел
        'min_samples_split': [5, 10, 20],  # увеличили минимум
        'min_samples_leaf': [2, 4, 8],  # увеличили минимум
        'ccp_alpha': [0.0, 0.01, 0.1]  # pruning
    },
    'R_random_forest': {
        'n_estimators': [100, 200, 300],  # расширили диапазон
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2', None],  # случайность признаков
        'bootstrap': [True],
        'oob_score': [True]  # OOB-оценка вместо CV для части контроля
    },
    'R_gradient_boosting_xg': {
        'n_estimators': [100, 200],  # больше базовых моделей
        'learning_rate': [0.01, 0.05, 0.1],  # более низкие LR
        'max_depth': [3, 4, 6],  # ограничили глубину
        'subsample': [0.8, 0.9],  # стохастичность
        'colsample_bytree': [0.8, 0.9, 1.0],  # случайный выбор признаков
        'reg_alpha': [0, 0.1, 1],  # L1-регуляризация
        'reg_lambda': [1, 5, 10]  # L2-регуляризация
    },
    'R_gradient_boosting_cat': {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [3, 5, 7],  # усиленная регуляризация
        'bagging_temperature': [0.8, 1.0],  # аналог subsample
        'random_strength': [1.0, 2.0]  # шум для устойчивости
    },
    'R_svm': {
        'C': [0.1, 0.5, 1, 5, 10],  # плотнее сетка регуляризации
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    },
    'R_knn': {
        'n_neighbors': [3, 5, 7, 9, 11],  # добавили больше соседей
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
        'p': [1, 2]  # порядок метрики Минковского
    },
    'C_logistic_regression': {
        'C': [0.1, 0.5, 1, 5, 10],  # плотнее сетка
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga'],
        'l1_ratio': [0.2, 0.5, 0.8]  # только для elasticnet
    },
    'C_decision_tree': {
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 4, 8],
        'criterion': ['gini', 'entropy'],
        'ccp_alpha': [0.0, 0.01, 0.1]
    },
    'C_random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True],
        'oob_score': [True]
    },
    'C_gradient_boosting_xg': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 6],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 5, 10]
    },
    'C_gradient_boosting_cat': {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [3, 5, 7],
        'bagging_temperature': [0.8, 1.0],
        'random_strength': [1.0, 2.0]
    },
    'C_svm': {
        'C': [0.1, 0.5, 1, 5, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'probability': [True]
    },
    'C_knn': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
        'p': [1, 2]
    }
}



# === НОВИНКА: СЛОВАРЬ ДЕЙСТВИЙ ===
# Этот словарь связывает "имя действия" (то, что выбирает пользователь)
# с "реальной функцией", которую нужно вызвать.
ACTION_DISPATCHER = {
    # --- Стандартные функции (вызывают сами себя) ---
    'data_overview': 'data_overview',
    'statistical_analysis': 'statistical_analysis',
    'distribution_visualization': 'distribution_visualization',
    'scatter_visualization': 'scatter_visualization',
    'correlation_analysis': 'correlation_analysis',
    'missing_values': 'missing_values',
    'unique_value_analysis': 'unique_value_analysis',
    'pivot_tables': 'pivot_tables',
    'deleting_columns': 'deleting_columns',
    'data_cleaning': 'data_cleaning',
    'оutlier_removal' : 'оutlier_removal',
    'duplicate_removal': 'duplicate_removal',
    'scaling_normalization' : 'scaling_normalization',
    'categorical_encoding': 'categorical_encoding',

    # --- Функции машинного обучения (все вызывают одну функцию) ---
    # Ключи — это все возможные имена моделей.
    **{name: 'machine_learning' for name in MODEL_MAP.keys()}
}



# Для сессий
from flask_session import Session
# from flask_cors import CORS
app = Flask(__name__)

# CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://127.0.0.1:5000"}})
# Что делает этот код:
#
# supports_credentials=True: Это самая важная часть. Она говорит браузеру: "Да, я разрешаю этому сайту отправлять куки (включая сессионные)".
# resources={r"/*": ...}: Мы применяем это правило ко всем маршрутам вашего приложения.
# origins="http://127.0.0.1:5000": Мы явно указываем адрес вашего сервера.

# Конфигурация секретного ключа и типа сессии (файловая для простоты)
app.secret_key = 'ваш_секретный_ключ_здесь_покруче'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)



UPLOAD_FOLDER = 'static/uploads'
import shutil
# Удаляем все содержимое папки
for filename in os.listdir(UPLOAD_FOLDER):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # удаляет файл или ссылку
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # удаляет папку
    except Exception as e:
        print(f'Ошибка при удалении {file_path}: {e}')
#ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx'}
ALLOWED_EXTENSIONS = {'csv', 'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

uploaded_filename = None
processed_actions = [] # Для хранения выбранных действий

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/choice')
def choice():
    # Передаем имя загруженного файла в шаблон
    return render_template('choice.html', filename=uploaded_filename)

@app.route('/result')
def result():
    # Отображает страницу результата, передавая выбранные действия
    return render_template('result.html', filename=uploaded_filename, actions=processed_actions)

@app.route('/upload-ajax', methods=['POST'])
def upload_ajax_file():
    global uploaded_filename
    global processed_actions # Сбрасываем действия при новой загрузке
    processed_actions = []

    if 'dataset_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'Нет поля для загрузки файла'}), 400
    file = request.files['dataset_file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Файл не выбран'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        uploaded_filename = filename

        return jsonify({
            'status': 'success',
            'message': f'Файл "{filename}" успешно загружен.',
            'redirect_url': url_for('choice')
        })
    else:
        return jsonify({'status': 'error', 'message': 'Недопустимый тип файла'}), 400

# маршрута для предпросмотра датасета
@app.route('/view-dataset', methods=['POST'])
def view_dataset():
    global uploaded_filename

    # Получаем данные из JSON тела запроса
    data = request.get_json(silent=True)

    # Если JSON не пришел (на всякий случай), пробуем получить из формы
    if not data:
        filename = request.form.get('filename')
        # Если и из формы не пришло, используем глобальное имя файла
        if not filename:
            filename = uploaded_filename
    else:
        filename = data.get('filename')
        # Если в JSON имя файла пустое, используем глобальное
        if not filename:
            filename = uploaded_filename

    # Проверки безопасности
    if not filename or not uploaded_filename:
        return jsonify({'status': 'error', 'message': 'Файл не найден'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Проверяем, что файл существует и его имя совпадает с глобальным (без пути)
    if not os.path.isfile(filepath) or secure_filename(filename) != secure_filename(uploaded_filename):
        return jsonify({'status': 'error', 'message': 'Файл не найден или не соответствует загруженному'}), 400

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath, sep=sepp) # sep=';'
        elif filename.endswith('.json'):
            df = pd.read_json(filepath)
        #elif filename.endswith('.xlsx'):
        #df = pd.read_excel(filepath)
        else:
            return jsonify({'status': 'error', 'message': 'Неподдерживаемый формат файла'}), 400

        if df.empty:
            return jsonify({'status': 'error', 'message': 'Файл пуст или не содержит данных'}), 400

        headers = df.columns.tolist()

        # Ограничиваем количество строк для предпросмотра, например, 10
        #preview_data = df.head(10).to_dict(orient='records')
        preview = df.head(5).replace({np.nan: None, np.inf: None, -np.inf: None}).values.tolist() # или вместо числа 15 написать None, тогда выведет всю таблицу

        return jsonify({
            'status': 'success',
            'headers': headers,
            'preview': preview
        })

    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return jsonify({'status': 'error', 'message': f'Ошибка чтения файла: {str(e)}'}), 500










"""Разведочный анализ данных"""

def data_overview(df): # Просмотр общей информации о датасете
    """
    Возвращает HTML-представление общей информации о DataFrame.
    Теперь работает корректно для любого DataFrame.
    """
    buffer = io.StringIO()

    # Перенаправляем вывод df.info() в буфер вместо консоли
    old_stdout = sys.stdout
    sys.stdout = buffer
    df.info()
    sys.stdout = old_stdout

    info_str = buffer.getvalue()

    # Оборачиваем в <pre> для сохранения форматирования и добавляем класс для стилизации

    result_html = f"""
            <h3>Обзор данных</h3>
            <pre class="code-block">{info_str}</pre>
            """

    return result_html

def statistical_analysis(df): # Статистический анализ числовых! признаков
    """
    Возвращает HTML-таблицу с базовой статистикой.
    """
    # to_html() возвращает строку с HTML-таблицей. Добавляем класс для стилизации CSS.
    result_html = f"""
                <h3>Статистический анализ</h3>
                <pre class="code-block">{df.describe().to_html(classes='data-table', index=True)}</pre>
                """

    return result_html

def distribution_visualization(df): # Визуализация распределений признаков (гистограммы, KDE, boxplot)
    """
    Визуализирует распределения числовых признаков: гистограмма, KDE, boxplot.
    Сохраняет графики в файл и возвращает HTML-код <img> для отображения.
    """
    # Выбираем только числовые признаки


    numeric_features = df.select_dtypes(include=['number']).columns

    # Если числовых признаков нет, возвращаем сообщение
    if len(numeric_features) == 0:
        return '<p class="error-message">В датасете нет числовых признаков для визуализации распределений.</p>'

    sns.set(style="whitegrid")

    # Определяем количество графиков и размер фигуры
    num_features = len(numeric_features)
    fig, axes = plt.subplots(num_features, 3, figsize=(9, 3 * num_features))

    # Если признак один, axes — одномерный массив, приводим к 2D
    if num_features == 1:
        axes = axes.reshape(1, 3)

    for i, feature in enumerate(numeric_features):
        # Гистограмма
        sns.histplot(df[feature], kde=False, ax=axes[i, 0])
        axes[i, 0].set_title(f'Гистограмма {feature}')

        # KDE
        sns.kdeplot(df[feature], ax=axes[i, 1], fill=True)
        axes[i, 1].set_title(f'KDE {feature}')

        # Boxplot
        sns.boxplot(x=df[feature], ax=axes[i, 2])
        axes[i, 2].set_title(f'Boxplot {feature}')

    plt.tight_layout()

    # Сохраняем изображение в статическую папку
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'distributions.png')
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    result_html = f"""
                    <h3 title="Гистограмма. Показывает частоту встречаемости значений в диапазонах.&#13;&#10KDE. Показывает плотность распределения данных. Площадь под всей кривой всегда равна 1.&#13;&#10Boxplot. Показывает основные статистики: медиану, квартели и выбросы. Центр — медиана, коробка — межквартильный диапазон, "усы" — минимальные и максимальные значения без выбросов, точки — выбросы.">Распределение признаков</h3>
                    <img src="{url_for("static", filename="uploads/distributions.png")}" alt="Визуализация распределений" class="analysis-image">
                    """

    return result_html

def scatter_visualization(df): # Визуализация взаимосвязей признаков (матрица диаграмм рассеяния)

    ### ДЛЯ ВСЕХ ПРИЗНАКОВ ### (не нужно принимать никакие параметры)
    plt.figure(figsize=(10, 10), dpi=70)
    sns.pairplot(df, kind="scatter") # это вернёт матрицу диаграмм рассеяния между всеми признаками # в будущем можно принимать от пользователя параметр hue, в котором название столбца относительно которого хотим диаграммы рассеяния

    # Сохраняем изображение в статическую папку
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'scatter_matrix.png')
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    # ### ДЛЯ ДВУХ ПРИЗНАКОВ ### (надо принять у пользователя col_x и col_y)
    # col_x = 'АД'
    # col_y = 'ИМТ'
    # plt.figure(figsize=(10, 6), dpi=90)
    # plt.scatter(df[col_x].values, df[col_y].values, c='blue', s=20) # принимает x, y, s - размер точек (маркеров) # это вернёт одну матрицу рассеяния между двумя выбранными признаками
    # plt.xlabel(col_x)
    # plt.ylabel(col_y)
    #
    # # Сохраняем изображение в статическую папку
    # img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'scatter.png')
    # plt.savefig(img_path, bbox_inches='tight')
    # plt.close()

    # result_html = f"""
    # <h5>Матрица диаграмм распределений для всех признаков</h5>
    # <img src="{url_for("static", filename="uploads/scatter_matrix.png")}" alt="Визуализация рассеяний всех признаков" class="analysis-image">
    # <h5>Диаграмма рассеяния для признаков "{col_x}" и "{col_y}"</h5>
    # <img src="{url_for("static", filename="uploads/scatter.png")}" alt="Визуализация рассеяний двух признаков" class="analysis-image">
    # """

    result_html = f"""
        <h3 title="Показывает взаимосвязь между двумя переменными.&#13;&#10Помогает выявить корреляцию между переменными и выбросы.">Диаграммы рассеяния</h3>
        <img src="{url_for("static", filename="uploads/scatter_matrix.png")}" alt="Визуализация рассеяний всех признаков" class="analysis-image">
        """

    return result_html

def correlation_analysis (df, params): # Анализ корреляций между числовыми! признаками
    """
    Анализ корреляций. Позволяет выбрать вывод: матрица, тепловая карта или оба варианта.
    params: {'output_type': 'matrix' | 'heatmap' | 'both'}
    """
    output_type = params.get('output_type', 'both') #  # по умолчанию — и то, и то

    corr = df.corr(numeric_only=True)  # корреляционная матрица # можно попробовать без параметра

    result_html = ""

    # 1. Выводим таблицу, если выбран 'matrix' или 'both'
    if output_type in ['matrix', 'both']:
        # Красивая таблица с классом для CSS
        result_html += f'<h3 title="Таблица, показывающая силу и направление линейной связи между всеми парами числовых столбцов.&#13;&#10Диапазон: от -1 (сильная обратная связь) до +1 (сильная прямая связь), 0 — отсутствие связи.">Корреляционная матрица</h3>'
        # result_html += corr.to_html(classes='data-table', index=True)
        result_html += f'<pre class="code-block">{corr.to_html(classes='data-table', index=True)}</pre>'

    # 2. Выводим тепловую карту, если выбран 'heatmap' или 'both'
    if output_type in ['heatmap', 'both']:
        plt.figure(figsize=(6, 4))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, square=True)
        # sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True) # без переменной cmap
        # plt.title('Матрица корреляций', fontsize=10)

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'correlation.png')
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()

        result_html += f'<h3 title="Цветовая визуализация корреляционной матрицы.">Тепловая карта корреляций</h3>'
        result_html += f'<img src="{url_for("static", filename="uploads/correlation.png")}" alt="Тепловая карта" class="analysis-image">'

    return result_html

def missing_values(df): # Проверка наличия пропущенных значений
    """
    Проверяет наличие пропущенных значений в каждом столбце DataFrame.
    Возвращает HTML-таблицу с количеством пропусков по каждому признаку.
    Если пропусков нет — возвращает информационное сообщение.
    """
    # Считаем количество пропусков по каждому столбцу
    missing = df.isnull().sum()

    """
    # Оставляем только те столбцы, где есть пропуски
    missing = missing[missing > 0]

    if missing.empty:
        return '<p>Пропущенные значения в датасете отсутствуют.</p>'
    """

    # Преобразуем в DataFrame для красивой табличной выдачи
    missing_df = missing.to_frame(name='Количество пропусков')
    missing_df.index.name = 'Признак'

    # Возвращаем HTML-таблицу с классом для стилизации
    #return missing_df.to_html(classes='data-table', justify='left') # получается некрасивая большая таблица

    result_html = f"""
            <h3>Проверка пропусков</h3>
            <pre class="code-block">{missing_df}</pre>
            """

    return result_html

def unique_value_analysis(df, params): # Анализ уникальных значений и частот
    """
    Анализирует количество уникальных значений и частоты для выбранных столбцов.
    params: {
        'columns': ['ИмяСтолбца1', 'ИмяСтолбца2'],
        'normalize': True или False  # false - абсолютные / true - относительные (%) частоты
    }
    """
    # Проверяем, что пришли нужные параметры
    columns = params.get('columns', [])

    # Получаем значение из словаря. Если его нет, по умолчанию считаем, что нужны абсолютные значения.
    raw_normalize_value = params.get('normalize', 'false')
    # Преобразуем строку в булево значение Python.
    # Только если строка в точности равна 'true', результат будет True. Во всех остальных случаях (включая 'false') — False.
    normalize = (raw_normalize_value == 'true')


    # Проверяем, что 'columns' является списком и этот список не пуст
    if not isinstance(columns, list) or len(columns) == 0:
        return '<p class="error-message">Не выбраны столбцы для анализа уникальных значений.</p>'

    # Проверяем, что все указанные столбцы есть в DataFrame
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        return f'<p class="error-message">Столбцы не найдены: {", ".join(missing_cols)}</p>'

    # 1. Блок: Количество уникальных значений для ВСЕХ столбцов
    # Собираем информацию о количестве уникальных значений
    unique_counts = df.nunique().to_frame(name='') # name='Уникальных значений' # dropna=True (по умолчанию) — NaN не учитываются при подсчёте
    #unique_counts.index.name = 'Признак'


    # 2. Блок: Частоты для ВЫБРАННЫХ столбцов
    freq_html_blocks = []
    for col in columns:
        vc = df[col].value_counts(dropna=False, normalize=normalize)
        vc_df = vc.to_frame(name='Частота')
        vc_df.index.name = 'Значение'

        # Определяем заголовок для блока частот
        freq_type = "Относительные (доли)" if normalize else "Абсолютные"
        freq_html_blocks.append(
            f'<h3>Частоты уникальных значений для "{col}" ({freq_type})</h3>' +
            f'<pre class="code-block">{vc_df.to_string()}</pre>'
        )

    # Собираем финальный HTML
    result_html = f"""
    <h3>Количество уникальных значений во всех столбцах</h3>
    <pre class="code-block">{unique_counts.to_string()}</pre>
    <hr>
    {''.join(freq_html_blocks)}
    """

    return result_html

def pivot_tables(df, params): # Создание сводных таблиц и группировок
    plt.figure(figsize=(6, 4))

    # row_categories = ['Диагноз']
    # col_categories = []
    # columns = ['Возраст']
    # functions = ['mean']

    """
    Создает сводную таблицу или группировку на основе параметров пользователя.
    params: {
        'mode': 'pivot' | 'group', 
        'row_categories': ['Столбец1'],
        'col_categories': ['Столбец2'],
        'columns': ['Столбец3'],
        'functions': ['mean']
    }
    """
    # Извлекаем параметры из словаря
    mode = params.get('mode', 'pivot')
    row_categories = params.get('row_categories', [])
    col_categories = params.get('col_categories', [])
    columns = params.get('columns', [])
    functions = params.get('functions', ['mean'])

    # Проверки на наличие обязательных данных
    if not row_categories or not columns:
        return '<p class="error-message">Для анализа необходимо выбрать хотя бы одну строку для группировки и один столбец для анализа.</p>'

    # Проверяем, что все указанные столбцы существуют в DataFrame
    all_cols = row_categories + col_categories + columns
    missing_cols = [c for c in all_cols if c not in df.columns]
    if missing_cols:
        return f'<p class="error-message">Столбцы не найдены: {", ".join(missing_cols)}</p>'

    result_html = ""

    # --- ГРУППИРОВКА (Group By) ---
    if mode == 'group':
        grouped = df.groupby(row_categories + col_categories)[columns].agg(functions)

        result_html += f'<h5>Группировка</h5>' # Группировка по: {", ".join(row_categories + col_categories)}
        # result_html += f'<h6>Агрегируемые столбцы: {", ".join(columns)}</h6>'
        # result_html += f'<h6>Функции: {", ".join(functions)}</h6>'
        result_html += f'<pre class="code-block">{grouped.to_string()}</pre>'
    # plt.title('Группировка', fontsize=10)


    # --- СВЁРТОЧНАЯ ТАБЛИЦА (Pivot Table) ---
    if mode == 'pivot':
        pivot = pd.pivot_table(
            df,
            index=row_categories, # колонки по которым группируем (станут строками итоговой таблицы)
            columns=col_categories, # колонки по которым группируем (станут столбцами сводной таблицы)
            values=columns, # колонки для анализа (столбцы, по которым считаются показатели)
            aggfunc=functions, # функции агрегации
            fill_value=0
        )

        result_html += f'<h3>Сводная таблица</h3>' # Сводная таблица (строки: {", ".join(row_categories)}
        # if col_categories:
        #     result_html += f'<h6>Столбцы: {", ".join(col_categories)}</h6>'
        # result_html += f'<h6>Значения: {", ".join(columns)}</h6>'
        # result_html += f'<h6>Функции: {", ".join(functions)}</h6>'
        # plt.title('Сводная таблица', fontsize=10)


        result_html += f'<pre class="code-block">{pivot.to_string()}</pre>'
    return result_html




"""Предварительная обработка данных"""

def feature_engineering(df): # Выделение и создание новых признаков : столбец (только числовой) операция (+,-,*,/,%...) столбец (только числовой)
    pass

def deleting_columns(df, params): # Удаление столбцов (например если в ней слишком много NaN или этот столбец не влияет на обучение)
    """
    Удаляет указанные столбцы из DataFrame.
    params: {'columns': ['ИмяСтолбца1', 'ИмяСтолбца2']}
    """
    col = params.get('columns', [])
    # col = ['АД', 'ИМТ']

    # Проверяем, что список столбцов не пуст
    if not col:
        return '<p class="error-message">Не указаны столбцы для удаления.</p>'

    # Проверяем, что все указанные столбцы есть в DataFrame
    missing_cols = [c for c in col if c not in df.columns]
    if missing_cols:
        return f'<p class="error-message">Столбцы не найдены: {", ".join(missing_cols)}</p>'


    # Удаляем столбцы (inplace=True — изменяем исходный DataFrame)
    df.drop(col, axis=1, inplace=True)
    # или может лучше возвращать новый df: return df.drop(col, axis=1) (при этом в любой момент могу вернуться к исходным, заного считав их в df из папки uploads)
    # или можно не менять исходные данные и не создавать новые, а просто создать новый csv/json в папке uploads , чтобы в любой момент пользоваться и исходными и изменёнными данными

    result_html = f"""
        <h3>Удалены столбцы : {", ".join([f'"{c}"' for c in col])}</h3>
        <h5>Оставшиеся столбцы : {", ".join([f'"{c}"' for c in df.columns.tolist()])}</h5>"""
    return result_html

def data_cleaning(df, params): # Очистка данных (от NaN)
    # params это dict вида {'column_name': 'action'}

    # Проверяем, есть ли вообще столбцы для обработки
    if not params:
        return '<p class="info-message">Действия по очистке данных не выбраны. Данные останутся без изменений.</p>'

    # Создаем копию названий столбцов, чтобы избежать ошибки при изменении df в цикле
    columns_to_process = list(params.keys())

    for col in columns_to_process:
        if col not in df.columns:
            continue

        action = params[col]
        col_dtype = df[col].dtype


        if action == 'заменить на 0':
            df[col] = df[col].fillna(0)
        elif action == 'заменить на среднее':
            df[col] = df[col].fillna(df[col].mean())
        elif action == 'заменить на медиану':
            df[col] = df[col].fillna(df[col].median())
        elif action == 'заменить на новую категорию unknown':
            df[col] = df[col].fillna('unknown')
        elif action == 'заменить на самое частое значение':
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
        elif action == 'предсказать значение с помощью KNNImputer':
            try:
                imputer = KNNImputer(n_neighbors=5)
                # Преобразуем столбец в 2D-массив для KNNImputer
                col_data = df[[col]].values
                imputed_data = imputer.fit_transform(col_data)
                df[col] = imputed_data.flatten()
                # или одной строкой
                # df[[col]] = imputer.fit_transform(df[[col]])
            except Exception as e:
                print(f"Ошибка KNNImputer: {e}")
        elif action == 'заполнить предыдущим значением':
            # Сначала заполняем вперёд, затем назад для оставшихся пропусков
            df[col] = df[col].fillna('ffill').fillna('bfill')
        elif action == 'заполнить следующим значением':
            # Сначала заполняем назад, затем вперёд для оставшихся пропусков
            df[col] = df[col].fillna('bfill').fillna('ffill')


    # Возвращаем первые 5 строк для предпросмотра результата
    # return f'<h5>Очистка данных завершена для столбцов: {", ".join(params.keys())}</h5>' + df.head(5).to_html(
    #     classes='data-table', index=False)
    return f'<h3>Очистка данных завершена для столбцов: {", ".join(params.keys())}</h3>' + missing_values(df)

def оutlier_removal(df, params):
    cols = params.get('columns', [])
    print(cols)
    if not cols:
        return '<p class="error-message">Не указаны столбцы для удаления выбросов.</p>'

    # Проверяем, что все указанные столбцы есть в DataFrame
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        return f'<p class="error-message">Столбцы не найдены: {", ".join(missing_cols)}</p>'

    # Сохраняем исходное количество строк для расчёта
    original_rows = len(df)

    # Вычисляем границы IQR для всех столбцов сразу на исходных данных
    bounds = {}
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR # 1.5 на 2.0 или 3.0 , чем больше число тем более экстремальные значения считаются выбросами и тем меньше строк удаляется
        upper_bound = Q3 + 3 * IQR
        bounds[col] = (lower_bound, upper_bound)

    # Создаём маску: строка остаётся, если все значения в пределах границ
    mask = pd.Series([True] * original_rows)
    for col, (lower, upper) in bounds.items():
        print(col)
        mask &= (df[col] >= lower) & (df[col] <= upper)

    # Применяем маску, изменяя исходный df на месте
    df.drop(df[~mask].index, inplace=True)

    # Рассчитываем и выводим количество удалённых строк
    removed_rows = original_rows - len(df)

    # result_html = f"""
    # <h5>Удалены выбросы у столбцов : {", ".join([f'"{c}"' for c in col])} !</h5>
    # <h5>Удалено строк: {removed_rows}</h5>
    # <h5>Осталось строк: {len(df)}</h5>""" + distribution_visualization(df)

    result_html = f"""
    <h3>Удалены выбросы у столбцов : {", ".join([f'"{c}"' for c in cols])} !</h3>
    <h5>Осталось строк: {len(df)} из {original_rows}</h5>""" #+ distribution_visualization(df)
    return result_html

def duplicate_removal(df): # Удаление дубликатов # сделать так, чтобы пользователь передавал столбцы, а если не передаст то по умолчанию будут все

    # 1. Считаем количество дубликатов до удаления
    duplicates_count_before = df.duplicated().sum()

    # 2. Считаем количество строк до удаления
    rows_before = len(df)

    # 3. Удаляем дубликаты
    df.drop_duplicates(inplace=True) # Удаление всех полных дубликатов (keep='first' по умолчанию)

    # col = [] # передаёт пользователь
    # df.drop_duplicates(subset=col, inplace=True) # Удаление дубликатов по определённым столбцам

    # df.drop_duplicates(keep='last', inplace=True) # Оставить первое/последнее вхождение дубликата (first / last)
    # Если данные упорядочены по времени (например, логи действий пользователя), то keep='last' может быть полезен, чтобы оставить самую свежую запись

    # 4. Считаем количество строк после удаления
    rows_after = len(df)

    # 5. Сколько строк удалено
    rows_removed = rows_before - rows_after

    # 6. Считаем количество дубликатов после удаления
    duplicates_count_after = df.duplicated().sum()

    # result_html = f"""
    # <h5>Количество дубликатов до удаления: {duplicates_count_before}</h5>
    # <h5>Строк удалено: {rows_removed}</h5>
    # <h5>Количество дубликатов после удаления: {duplicates_count_after}</h5>
    # """
    # return result_html

    result_html = f"""
        <h3>Удалено дубликатов: {rows_removed} из {duplicates_count_before}</h3>
        <h5>Осталось строк: {len(df)} из {rows_before}</h5>"""
    return result_html

    # или в одну строку
    # return f'<h5>Количество дубликатов до удаления: {duplicates_count_before}</h5> <h5>Строк удалено: {rows_removed}</h5> <h5>Количество дубликатов после удаления: {duplicates_count_after}</h5>'

def scaling_normalization(df, params): # нормализация числовых признаков
    """
    params: {'output_type': 'Min-Max' | 'StandardScaler' | 'RobustScaler' | 'MaxAbsScaler'}
    """
    first_col = df.iloc[:, 0]  # берём первый столбец по индексу
    range_min = first_col.min()
    range_max = first_col.max()
    print(f"Диапазон первого столбца: от {range_min} до {range_max}")

    numeric_features = df.select_dtypes(include=['number']).columns

    # Если числовых признаков нет, возвращаем сообщение
    if len(numeric_features) == 0:
        return '<p class="error-message">В датасете нет числовых признаков для визуализации распределений.</p>'

    output_type = params.get('output_type', 'Min-Max') #  # по умолчанию

    if output_type == 'Min-Max':
        scaler = MinMaxScaler()
    elif output_type == 'StandardScaler':
        scaler = StandardScaler()
    elif output_type == 'RobustScaler':
        scaler = RobustScaler()
    elif output_type == 'MaxAbsScaler':
        scaler = MaxAbsScaler()

    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    first_col = df.iloc[:, 0]  # берём первый столбец по индексу
    range_min = first_col.min()
    range_max = first_col.max()
    print(f"Диапазон первого столбца: от {range_min} до {range_max}")

    return f'<h3>Нормализация успешно произведена!</h3>'

def categorical_encoding(df, params): # Кодирование категориальных признаков (преобразует текстовые категории в числовой формат) (у меня Ordinal encoding (ещё называют Label Encoding) (порядковое кодирование) а не One-hot)
    """
    Кодирует категориальные признаки в указанных столбцах, заменяя категории на числа.
    Изменяет исходный DataFrame и возвращает первые 5 строк в виде HTML-таблицы.
    """

    categorical_columns = params.get('columns', [])
    # Проверяем, что список столбцов не пуст
    if not categorical_columns:
        return '<p class="error-message">Не указаны столбцы для кодирования.</p>'

    # Проверяем, что все указанные столбцы есть в DataFrame
    # missing_cols = [c for c in categorical_columns if c not in df.columns]
    # if missing_cols:
    #     return f'<p class="error-message">Столбцы не найдены: {", ".join(missing_cols)}</p>'

    # Для каждого столбца из списка выполняем кодирование
    for column in categorical_columns:
        # Оставляем только уникальные значения, сохраняя порядок появления
        unique_values = df[column].dropna().unique()
        # Создаём словарь для замены: категория -> число (начиная с 1)
        mapping = {value: i + 1 for i, value in enumerate(unique_values)}
        # Заменяем категории на числа (NaN остаются как есть)
        df[column] = df[column].map(mapping)



    ### One-Hot Encoding на всякий случай
    # pd.get_dummies(df[col], drop_first=True)
    # Создает новые бинарные(0 / 1) столбцы для каждой уникальной категории в указанных переменных.
    # Для каждого уникального значения в категориальных колонках он создает отдельный столбец.
    # В ячейке стоит 1, если исходное значение равно этой категории, и 0 — иначе.
    # drop_first=True Удаляет один из созданных столбцов для каждой категории, чтобы избежать проблемы мультиколлинеарности(зависимости между признаками)


    # Возвращаем первые 5 строк в виде HTML-таблицы
    return f'<h3>Закодировы столбцы : {', '.join([f'"{c}"' for c in categorical_columns])}</h3>' + df.head(5).to_html(classes='data-table', index=False)
    # return f'<h5>Столбцы {', '.join([f'"{c}"' for c in categorical_columns])} закодированы!</h5>'

def time_series(df): # Работа с временными рядами (сдвиги, скользящие окна)
    pass




"""Построение моделей и машинное обучение"""

def machine_learning(df, model_name, params):
    """
    Обучает и оценивает модель машинного обучения.
    Параметры:
        df: DataFrame с данными.
        model_name: Ключ из MODEL_MAP (например, 'R_linear_regression').
        params: 'target_col'.
    """
    target_col = params.get('target_col')

    # Проверка наличия целевой колонки
    if not target_col or target_col not in df.columns:
        return f'<p class="error-message">Целевая колонка не выбрана или не найдена в данных.</p>'

    # Проверка наличия модели в словаре
    if model_name not in MODEL_MAP:
        return f'<p class="error-message">Модель "{model_name}" не найдена в списке доступных.</p>'


    # Считаем количество пропусков по каждому столбцу
    missing = df.isnull().sum().sum()

    if missing != 0:
        print(missing)
        return f'<p class="error-message">В признаках есть пропуски. Выполните очистку данных от пропусков перед обучением модели.</p>'

    # --- НОВЫЙ БЛОК ДЛЯ СОЗДАНИЯ МОДЕЛИ ---

    # 1. Получаем класс модели из словаря
    # model_class = MODEL_MAP[model_name]


    model_class = MODEL_MAP[model_name]['model']

    # Выделение зависимой (целевой) переменной y
    y = df[target_col]
    # Если задача классификации, и в целевой переменной есть строки — кодируем # кодирует только целевой столбец, так что если есть ещё категориальные столбцы то будет ошибка
    # if model_name.startswith('C_'):
    #     # Преобразуем в числовой формат (Label Encoding)
    #     y = y.astype('category').cat.codes

    X = df.drop(columns=[target_col])


    # Проверка на наличие только числовых признаков (для большинства моделей)
    if not all(dtype.kind in 'bifc' for dtype in X.dtypes):
        return '<p class="error-message">В признаках есть нечисловые столбцы. Выполните кодирование категориальных признаков перед обучением модели.</p>'

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 2. Создаем экземпляр модели (объект)
    # Для большинства моделей мы хотим, чтобы результаты были воспроизводимы,
    # поэтому передаем random_state=0 в конструктор.
    # Для SVC (C_svm) обязательно нужно probability=True, иначе не будут работать вероятности классов.



    if model_name == 'C_svm':
        base_model = model_class(random_state=0, probability=True)
    elif model_name in ['C_knn', 'R_knn', 'R_linear_regression', 'R_svm']:
        base_model = model_class()
    else:
        base_model = model_class(random_state=0)


    # 4. Создаём GridSearch с кросс‑валидацией
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grids[model_name],
        cv=5,  # 5‑fold кросс‑валидация
        scoring='neg_mean_squared_error' if model_name.startswith('R_') else 'accuracy',
        n_jobs=-1  # используем все ядра процессора
    )

    # Вместо GridSearchCV создаём RandomizedSearchCV:
    # random_search = RandomizedSearchCV(
    #     estimator=base_model,
    #     param_distributions=param_grids[model_name],  # используем те же сетки
    #     n_iter=20,  # количество случайных комбинаций
    #     cv=5,
    #     scoring='neg_mean_squared_error' if model_name.startswith('R_') else 'accuracy',
    #     random_state=0,
    #     n_jobs=-1
    # )




    try:
        grid_search.fit(X_train, y_train) # 5. Обучаем GridSearch (подбираем лучшие параметры)
        model = grid_search.best_estimator_ # 6. Берём лучшую модель

        # random_search.fit(X_train, y_train)
        # model = random_search.best_estimator_

        # base_model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    except Exception as e:
        return f'<p class="error-message">Ошибка при обучении модели: {str(e)}</p>'


    # --- ДОБАВЛЕННЫЙ КОД: СОХРАНЕНИЕ МОДЕЛИ ---



    # Генерируем уникальное имя файла
    model_filename = f"model_{model_name}_{uuid.uuid4().hex[:8]}.pkl"
    model_filepath = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)

    # Сохраняем модель
    joblib.dump(model, model_filepath)

    # Формируем HTML с кнопкой скачивания

    result_html = f'''
    <h3 style="margin: 0; padding-right: 30px; position: relative;">
      Модель: {MODEL_MAP[model_name]['name']}
      <a href="/download-model?filename={model_filename}" title='Скачать обученную модель'
         style="position: absolute; top: 0; right: 0; font-size: 20px; color: #1e2129; text-decoration: none;">
        &#8681;
      </a>
    </h3>
    '''

    if model_name.startswith('R_'): # Метрики для регрессии
        err_1 = mean_absolute_error(y_test, y_pred) # или просто np.mean(np.abs(y_test - y_pred)) # тогда и метод mean_absolute_error в импортах не нужен
        err_2 = np.sqrt(mean_squared_error(y_test, y_pred))
        result_html += f"""
            <p title="Средняя абсолютная ошибка.&#13;&#10Это средняя величина ошибки предсказаний, без акцента на выбросы (просто берёт модуль всех отклонений и усредняет их).&#13;&#10Диапазон: [0;+∞). Чем меньше, тем лучше.&#13;&#10Менее чувствительна к выбросам.&#13;&#10Используется, чтобы показать «честную» среднюю ошибку, не искажённую выбросами">MAE: {err_1:.4f}</p>
            <p title="Корень из среднеквадратичной ошибки.&#13;&#10Это среднее «расстояние» между предсказанными и реальными значениями, но с усиленным акцентом на большие ошибки (сначала возводит ошибки в квадрат (усиливая влияние больших промахов), усредняет, а затем извлекает корень).&#13;&#10Диапазон: [0;+∞). Чем меньше, тем лучше.&#13;&#10Чувствительна к выбросам: если модель сильно ошиблась хотя бы несколько раз, RMSE заметно вырастет.&#13;&#10Используется, чтобы подсветить риск крупных ошибок — критично для финансов, медицины и т.п.">RMSE: {err_2:.4f}</p>
        """

    else:
        err_1 = accuracy_score(y_test, y_pred)
        err_2 = confusion_matrix(y_test, y_pred)

        classes = sorted(set(y_test)) # pd.Series(y_test).unique() # определяем названия классов
        df_cm = pd.DataFrame(err_2, index=classes, columns=classes) # Создаем DataFrame с подписями строк и столбцов
        html_table = df_cm.to_html(index=True, border=1)


        result_html += f"""
        <p title="Доля правильных предсказаний среди всех.&#13;&#10Диапазон: [0;1]. Чем больше, тем лучше.&#13;&#10Может быть обманчивой при дисбалансе классов.">Accuracy: {err_1:.4f}</p>
        <p title="Матрица ошибок.&#13;&#10Диапазон значений в ячейках: [0;𝑁] , где 𝑁 — общее число наблюдений.&#13;&#10Главная диагональ — верные ответы. Остальные ячейки — ошибки.">Confusion Matrix:</p>
        <div style="text-align: center; margin-bottom: 3px;">
            <span style="font-size: small;">Предсказание</span>
        </div>
        <div style="text-align: center; margin-bottom: 3px; margin-right: 22px;">
            <table style="margin: 0 auto; border-collapse: collapse; font-size: small">
                <tr>
                    <td style="vertical-align: middle; text-align: center; writing-mode: vertical-rl; transform: rotate(180deg); padding: 3px;">
                        Истина
                    </td>
                    <td>
                        {html_table}
                    </td>
                </tr>
            </table>
        </div>
        """

    #f"""< p > Целевая колонка: < strong > {target_col} < / strong > < / p >"""
    return result_html






@app.route('/start-process', methods=['POST'])
def start_process(): # очищает сессию и запускает процесс
    # === НОВЫЕ СТРОКИ ДЛЯ ОТЛАДКИ ===
    print("--- ЗАПРОС НА СТАРТ ПРОЦЕССА ПОЛУЧЕН ---")
    data = request.get_json()
    print("JSON ТЕЛО ЗАПРОСА:", data)
    print("--- КОНЕЦ ЛОГОВ ---")


    # Сбрасываем сессию при каждом новом запуске
    session.clear()

    filename = request.json.get('filename')
    actions = request.json.get('actions')

    if not filename or not actions:
        return jsonify({'status': 'error', 'message': 'Нет данных'}), 400

    # Сохраняем в сессию имя файла и список действий
    session['uploaded_filename'] = filename
    session['actions_queue'] = actions.copy()  # Копия, чтобы не менять оригинал
    session['current_df'] = None  # Здесь будем хранить DataFrame после изменений


    # Переходим к первому шагу
    return get_next_action()

@app.route('/get-next-action', methods=['GET'])
def get_next_action(): # будет возвращать либо форму для ввода параметров, либо результат.
    if 'actions_queue' not in session or not session['actions_queue']: # if not session.get('actions_queue') or len(session['actions_queue']) == 0:
        # Все действия выполнены
        return jsonify({'status': 'finished'})

    # Берем первое действие из очереди
    next_action = session['actions_queue'][0]

    # Загрузка DataFrame для получения списка столбцов (если его еще нет в сессии)
    if session.get('current_df') is None:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_filename'])
        try:
            if filepath.endswith('.csv'):
                df_for_preview = pd.read_csv(filepath, sep=sepp, nrows=1) # Читаем только заголовки! sep=';'
            elif filepath.endswith('.json'):
                df_for_preview = pd.read_json(filepath)
            else:
                return jsonify({'status': 'error', 'message': 'Неподдерживаемый формат'}), 400
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Ошибка чтения файла для предпросмотра: {e}'}), 500
    else:
        df_for_preview = session['current_df']




    # Логика для действий, требующих ввода параметров от пользователя

    if next_action == 'correlation_analysis':
        # Формируем схему для выбора типа вывода (Радиокнопки)
        return jsonify({
            'status': 'need_params',
            'action': next_action,
            'params_schema': {
                'type': 'radio',
                'label': 'Выберите тип вывода корреляционного анализа:',
                'options': [
                    { 'value': 'matrix', 'label': 'Только корреляционная матрица' },
                    { 'value': 'heatmap', 'label': 'Только тепловая карта' },
                    { 'value': 'both', 'label': 'Оба варианта (матрица + карта)' }
                ],
                'name': 'output_type'
            }
        })

    elif next_action == 'unique_value_analysis':
        columns_list = df_for_preview.columns.tolist();

        return jsonify({
            'status': 'need_params',
            'action': next_action,
            'params_schema': {
                'type': 'complex', # Тип формы с радиокнопками и чекбоксами
                'columns_list': columns_list,
                'label': 'Анализ уникальных значений и частот'
            }
        })

    elif next_action == 'pivot_tables':
        columns_list = df_for_preview.columns.tolist()

        return jsonify({
            'status': 'need_params',
            'action': next_action,
            'params_schema': {
                'type': 'complex_pivot',
                'columns_list': columns_list,
                'label': 'Настройка сводной таблицы или группировки'
            }
        })

    elif next_action == 'deleting_columns':
        # Формируем список столбцов для передачи во фронтенд
        columns_list = df_for_preview.columns.tolist()

        return jsonify({
            'status': 'need_params',
            'action': next_action,
            'params_schema': {
                'type': 'checkbox',
                'label': 'Выберите столбцы для удаления:',
                'options': columns_list,
                'name': 'columns'
            }
        })

    elif next_action == 'data_cleaning':
        # Загружаем DataFrame для получения типов столбцов
        if session.get('current_df') is None:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_filename'])
            try:
                if filepath.endswith('.csv'):
                    df_for_preview = pd.read_csv(filepath, sep=sepp, nrows=100)
                elif filepath.endswith('.json'):
                    df_for_preview = pd.read_json(filepath)
                else:
                    return jsonify({'status': 'error', 'message': 'Неподдерживаемый формат'}), 400
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'Ошибка чтения файла для предпросмотра: {e}'}), 500
        else:
            df_for_preview = session['current_df']
        columns_list = df_for_preview.columns.tolist()
        column_types = {col: str(df_for_preview[col].dtype) for col in columns_list}
        # Формируем полную схему с опциями для каждого столбца
        column_options = []
        for col in columns_list:
            col_type = str(df_for_preview[col].dtype)
            if col_type in ['int64', 'float64']:
                actions = [
                    "заменить на 0",
                    "заменить на среднее",
                    "заменить на медиану",
                    "предсказать значение",
                    "заполнить предыдущим значением",
                    "заполнить следующим значением"
                ]

            else:
                actions = [
                    "заменить на новую категорию unknown",
                    "заменить на самое частое значение",
                    "заполнить предыдущим значением",
                    "заполнить следующим значением"
                ]

            column_options.append({
                'column_name': col,
                'column_type': col_type,
                'actions': actions
            })

        # Создаём params_schema ПОСЛЕ того, как сформировали column_options
        params_schema = {
            'type': 'cleaning_table',
            'columns': column_options,
            'label': 'Выберите метод обработки пропусков:'
        }

        # Теперь можно безопасно вывести отладочную информацию
        print(f"--- DEBUG: DATA_CLEANING SCHEMA ---")
        print(f"Columns: {columns_list}")
        print(f"Column types: {column_types}")
        print(f"Schema to be sent: {params_schema}")
        print("--- END DEBUG ---")
        return jsonify({
            'status': 'need_params',
            'action': next_action,
            'filename': session['uploaded_filename'],
            'params_schema': params_schema
        })

    elif next_action == 'оutlier_removal':
        columns_list = df_for_preview.select_dtypes(include=['number']).columns.tolist() # 'int64', 'float64'


        return jsonify({
            'status': 'need_params',
            'action': next_action,
            'params_schema': {
                'type': 'checkbox',
                'label': 'Выберите числовые столбцы для удаления выбросов:',
                'options': columns_list,
                'name': 'columns'
            }
        })

    elif next_action == 'scaling_normalization':
        return jsonify({
            'status': 'need_params',
            'action': next_action,
            'params_schema': {
                'type': 'radio',
                'label': 'Выберите способ нормализации:',
                'options': [
                    { 'value': 'Min-Max', 'label': 'Min-Max Scaling' },
                    { 'value': 'StandardScaler', 'label': 'StandardScaler' },
                    {'value': 'RobustScaler', 'label': 'RobustScaler'},
                    {'value': 'MaxAbsScaler', 'label': 'MaxAbsScaler'}
                ],
                'name': 'output_type'
            }
        })

    elif next_action == 'categorical_encoding':
        columns_list = df_for_preview.select_dtypes(include=['object', 'category', 'str']).columns.tolist()

        return jsonify({
            'status': 'need_params',
            'action': next_action,
            'params_schema': {
                'type': 'checkbox',
                'label': 'Выберите категориальные столбцы для кодирования:',
                'options': columns_list,
                'name': 'columns'
            }
        })

    elif next_action.startswith('R_') or next_action.startswith('C_'):
        # Это действие машинного обучения

        columns_list = df_for_preview.columns.tolist()

        # Получаем русское название модели
        model_info = MODEL_MAP.get(next_action)
        if not model_info:
            return jsonify({'status': 'error', 'message': 'Модель не найдена'})

        return jsonify({
            'status': 'need_params',
            'action': next_action,
            'params_schema': {
                'type': 'complex_ml',
                'columns_list': columns_list,
                'model_name': next_action,  # ключ для внутреннего использования
                'display_name': model_info['name'],  # русское название для отображения
            }
        })

    # Для действий без параметров (например, data_overview)
    else:
        return jsonify({
            'status': 'execute_now',
            'action': next_action,
            'params': {}
        })

@app.route('/submit-action-params', methods=['POST'])
def submit_action_params(): # Сюда будет приходить POST-запрос с данными формы.
    data = request.json
    action_name = data.get('action')
    params_from_client = data.get('params', {})


    # Специальная обработка для сложной формы data_cleaning
    if action_name == 'data_cleaning':
        params_from_client_cleaned = {}
        for key, value in params_from_client.items():
            if key.startswith('cleaning_') and value:  # проверяем, что метод выбран (не пустая строка)
                column_name = key.replace('cleaning_', '')
                params_from_client_cleaned[column_name] = value
        # теперь передаём в функцию только отфильтрованный словарь
        params_from_client = params_from_client_cleaned
        if not params_from_client_cleaned:
            # если ни один столбец не выбран для очистки, возвращаем сообщение
            return jsonify({
                'status': 'success',
                'result_html': '<p class="info-message">Ни один столбец не был выбран для очистки. Данные остались без изменений.</p>',
                'next_action': get_next_action()  # сразу переходим к следующему действию
            })

    # Загрузка DataFrame (из сессии или с диска)
    if session.get('current_df') is None:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_filename'])
        try:
            if filepath.endswith('.csv'):
                df_main = pd.read_csv(filepath, sep=sepp) # sep=';'
            elif filepath.endswith('.json'):
                df_main = pd.read_json(filepath)
            else:
                return jsonify({'status': 'error', 'message': f'Неподдерживаемый формат файла для {action_name}.'}), 400
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Критическая ошибка чтения файла: {e}'}), 500
    else:
        df_main = session['current_df']

    try:
        # Создаем копию DataFrame, чтобы не испортить оригинал на случай ошибки в функции
        df_working_copy = df_main.copy()


        # func_to_call = globals()[action_name]
        # --- НАЧАЛО ЗАМЕНЫ ---
        # Получаем "реальное" имя функции из нашего словаря-диспетчера
        real_function_name = ACTION_DISPATCHER.get(action_name)

        if not real_function_name:
            return jsonify({'status': 'error', 'message': f'Действие {action_name} не найдено в диспетчере.'}), 500

        # Теперь получаем саму функцию по её реальному имени
        func_to_call = globals().get(real_function_name)

        if not func_to_call:
            return jsonify({'status': 'error',
                            'message': f'Функция {real_function_name} не найдена в глобальном пространстве имен.'}), 500
        # --- КОНЕЦ ЗАМЕНЫ ---

        # --- НОВАЯ ЛОГИКА ВЫЗОВА ---

        # 1. Проверяем, является ли функция - функцией машинного обучения
        if func_to_call.__name__ == 'machine_learning':
            # Для МО-функции нужно 3 аргумента: df, имя_модели, параметры
            # Создаем копию параметров, чтобы не менять оригинал
            final_params = params_from_client.copy()
            final_params['model_name'] = action_name  # Добавляем имя модели в параметры
            print(f"--- ОТПРАВЛЯЕМЫЕ ПАРАМЕТРЫ В {action_name} ---")
            print(f"action_name (модель): {action_name}")
            print(f"params_from_client (от пользователя): {params_from_client}")
            print(f"final_params (после добавления model_name): {final_params}")
            print(f"Тип объекта params: {type(final_params)}")
            print("--- КОНЕЦ ЛОГОВ ---")
            result_html_str = func_to_call(df_working_copy, action_name, final_params)

        # 2. Если это не МО, проверяем количество параметров для остальных функций
        else:
            sig = inspect.signature(func_to_call)
            params_count = len(sig.parameters)

            if params_count == 1:
                # Функция принимает только DataFrame (например, data_overview)
                result_html_str = func_to_call(df_working_copy)
            elif params_count == 2:
                # Функция принимает DataFrame и параметры (например, deleting_columns)
                result_html_str = func_to_call(df_working_copy, params_from_client)
            else:
                # На случай, если у какой-то другой функции окажется 3+ параметра
                return jsonify(
                    {'status': 'error', 'message': f'Неподдерживаемая сигнатура функции {func_to_call.__name__}.'}), 500

        # --- КОНЕЦ НОВОЙ ЛОГИКИ ---


        # Сохраняем измененный DataFrame в сессию для следующего шага!
        session['current_df'] = df_working_copy

        # Удаляем выполненное действие из очереди
        session['actions_queue'].pop(0)

        return jsonify({
            'status': 'success',
            'result_html': result_html_str,
            # Сразу запрашиваем следующий шаг, чтобы фронтенд не делал лишний запрос?
            # Или нет, чтобы показать кнопку "Далее". Лучше нет.
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Ошибка при выполнении {action_name}: {str(e)}'}), 500

@app.route('/get-column-types', methods=['POST'])
def get_column_types():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({'status': 'error', 'message': 'Не указан файл'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath, sep=sepp, nrows=100)  # Читаем часть данных для определения типов
        elif filename.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            return jsonify({'status': 'error', 'message': 'Неподдерживаемый формат'}), 400

        column_types = {col: str(df[col].dtype) for col in df.columns}

        print(f"--- DEBUG: GET_COLUMN_TYPES ---")
        print(f"Filename: {filename}")
        print(f"Column types: {column_types}")
        print("--- END DEBUG ---")

        return jsonify({'status': 'success', 'column_types': column_types})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Ошибка чтения файла: {str(e)}'}), 500

@app.route('/download-model')
def download_model():
    filename = request.args.get('filename')
    if not filename:
        return "Файл не указан", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "Файл модели не найден", 404



    return send_file(
        filepath,
        as_attachment=True,
        download_name=filename
    )


if __name__ == '__main__':
    app.run(debug=True)
