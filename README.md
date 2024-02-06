# 🏦 Задача предсказания отклика клиентов банка
[![readme.jpg](https://anopic.ag/rBH1eeLq0d0KDZ2pPIpatAoMAnxYjQZ37jl3MFmf.jpg)](https://anopic.ag/rBH1eeLq0d0KDZ2pPIpatAoMAnxYjQZ37jl3MFmf.jpg)

*Веб-сервис с разведочным анализом данных доступен [здесь](https://bankcustomers.streamlit.app/)*

## Описание проекта
Один из способов повышения эффективности взаимодействия банка с клиентами заключается в том, чтобы отправлять предложение о новой услуге не всем клиентам банка, а только некоторой части, выбираемой по принципу наибольшей склонности к отклику на данное предложение.

Конкурсное задание заключается в том, чтобы предложить алгоритм, который будет выдавать оценку склонности клиента к положительному отклику по его признаковому описанию. Эта оценка может (хотя и не обязана) интерпретироваться как вероятность положительного отклика. Предполагается, что, получив такие оценки для некоторого множества клиентов, банк обратится с предложением только к тем клиентам, у которых значение оценки выше некоторого порога.

## Содержание

- [🏦 Задача предсказания отклика клиентов банка](#-задача-предсказания-отклика-клиентов-банка)
  - [Описание проекта](#описание-проекта)
  - [Содержание](#содержание)
  - [Файлы](#файлы)
  - [Данные](#данные)
  - [Разведочный анализ данных](#разведочный-анализ-данных)
  - [ML модель](#ml-модель)
  - [Разработка](#разработка)
  - [Как установить и запустить проект](#как-установить-и-запустить-проект)
  - [Функциональность приложения](#функциональность-приложения)
  - [Автор](#автор)
  - [Лицензия](#лицензия)

## Файлы

- [eda.ipynb](https://github.com/moxeeem/BankCustomerResponse/blob/main/eda.ipynb) : Jupyter Notebook с разведочным анализом данных
- [/datasets](https://github.com/moxeeem/BankCustomerResponse/tree/main/datasets) : Исходные датасеты
- [/app](https://github.com/moxeeem/BankCustomerResponse/blob/main/app) : Исходный код приложения (frontend и backend)


## Данные
В данном проекте мы работаем с [базой данных](https://github.com/aiedu-courses/stepik_linear_models/tree/main/datasets/clients), которая хранит информацию о клиентах банка и их персональные данные, такие как пол, количество детей и другие.

Исходная база данных была вынесена на публичный сервис. 

После склейки таблиц мы получили датасет со следующими колонками:

| Колонка | Описание | Тип данных |
|----------------------------|----------------------------------|------------|
| `AGREEMENT_RK` | уникальный идентификатор объекта в выборке | int64 |
| `AGE` | возраст клиента | int64 |
| `GENDER` | пол клиента (1 — мужчина, 0 — женщина) | int64 |
| `EDUCATION` | образование | object |
| `MARITAL_STATUS` | семейное положение | object |
| `CHILD_TOTAL` | количество детей клиента | int64 |
| `DEPENDANTS` | количество иждивенцев клиента | int64 |
| `SOCSTATUS_WORK_FL` | социальный статус клиента относительно работы (1 — работает, 0 — не работает) | int64 |
| `SOCSTATUS_PENS_FL` | социальный статус клиента относительно пенсии (1 — пенсионер, 0 — не пенсионер) | int64 |
| `REG_ADDRESS_PROVINCE` | область регистрации клиента | object |
| `FACT_ADDRESS_PROVINCE` | область фактического пребывания клиента | object |
| `POSTAL_ADDRESS_PROVINCE` | почтовый адрес области | object |
| `FL_PRESENCE_FL` | наличие в собственности квартиры (1 — есть, 0 — нет) | int64 |
| `OWN_AUTO` | количество автомобилей в собственности | int64 |
| `GEN_INDUSTRY` | отрасль работы клиента | object |
| `GEN_TITLE` | должность | object |
| `JOB_DIR` | направление деятельности внутри компании | object |
| `WORK_TIME` | время работы на текущем месте (в месяцах) | float64 |
|`FAMILY_INCOME`|семейный доход (несколько категорий)|object|
|`PERSONAL_INCOME`|личный доход клиента (в рублях)|float64|
|`CREDIT`|сумма последнего кредита клиента (в рублях)|float64|
|`TERM`|срок кредита|int64|
|`FST_PAYMENT`|первоначальный взнос (в рублях)|float64|
|`LOAD_NUM_TOTAL`|количество ссуд клиента|int64|
|`LOAD_NUM_CLOSED`|количество погашенных ссуд клиента|int64|
|**`TARGET`**|отклик на маркетинговую кампанию (1 — отклик был зарегистрирован, 0 — отклика не было)|int64|



## Разведочный анализ данных

В рамках разведочного анализа данных мы очистили данные от дубликатов и заполнили пропущенные значения. Также были построены графики распределений признаков, графики корреляций и графики зависимости целевой переменной от признаков. Кроме того, мы вычислили числовые характеристики столбцов и провели статистические тесты.

Разведочный анализ данных мы развернули в виде веб-сервиса с использованием библиотеки Streamlit.

## ML модель

В данном проекте мы использовали модели LogisticRegression и SVM, также подбирали гиперпараметры. Перед созданием модели мы закодировали данные при помощи OneHotEncoder, масштабировали их при помощи StandartScaler.

В данном проекте наилучшие результаты показала модель логистической регрессии со стандартными гиперпараметрами. 

Метрики линейной регрессии при стандартном пороге (0.5):

|Метрика|Значение|
|-------|--------|
|Accuracy|0.8854|
|Precision|1|
|Recall|0.0057|
|F1-score|0.0113|
|ROC-AUC|0.5028|



## Разработка

Для начала, данные были перемещены на публичный сервер Neon.tech.

Затем был разработан веб-сервис на FastAPI для извлечения данных из хранилища и предоставления эндпоинтов для их обработки. Этот веб-сервис также предоставляет возможности обучения моделей машинного обучения и получения прогнозов от них. Реализованный веб-сервис был развернут на публичном сервере Render.

Кроме того, было создано приложение Streamlit, которое предоставляет разведочный анализ данных, результаты обучения моделей и индивидуальные прогнозы для клиентов банка.


## Как установить и запустить проект

Исходные файлы приложения находятся в папке [/app](https://github.com/moxeeem/BankCustomerResponse/blob/main/app).

Для осуществления локального запуска приложения требуется установить все зависимости, перечисленные в файле requirements.txt, при помощи менеджера пакетов pip.

Frontend приложения запускается при помощи команды:

`streamlit run frontend/app.py`

Backend приложения запускается при помощи команды:

`uvicorn backend.app:app --reload`


## Функциональность приложения

Приложение предоставляет полный разведочный анализ данных, визуализацию результатов моделей машинного обучения с подбором гиперпараметров и индивидуальными прогнозами, а также статистические тесты.

Скриншот раздела "ML модель":

<img src="http://anopic.ag/MoKU0lb3cBP7TKYH8Z5g1yR2nAnMwnSuJpfV73d3.png" width="500">

Скриншоты разделов, связанных с разведочным анализом данных:

<img src="http://anopic.ag/GbzgoQ4FQ7AS1rPrINndGpadSbQjR62DXs2svzVg.png" width="500">


<img src="http://anopic.ag/RPbogCqLAaQLD5KEzXvNfhnmkVBbgArhNUSkwPt2.png" width="500">


<img src="http://anopic.ag/HCSMHNJ3CQ0VfJGnfHQU8mmdvl6LOUJ4xKIKFDOA.png" width="500">


## Автор
- Максим Иванов - [GitHub](https://github.com/moxeeem), [Telegram](https://t.me/fwznn_ql1d_8)

Данный репозиторий создан в рамках курса ["Линейные модели и их презентация"](https://stepik.org/course/177215) от [AI Education](https://stepik.org/users/628121134).

## Лицензия
Данный репозиторий лицензируется по лицензии MIT. Дополнительную информацию см. в файле [LICENSE](/LICENSE).
