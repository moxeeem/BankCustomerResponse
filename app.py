import streamlit as st
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import plotly.graph_objects as go
from PIL import Image
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu
from eda import open_data, drop_duplicates, concat_data, fill_nans
st.set_option('deprecation.showPyplotGlobalUse', False)


def preload_content():
    df1, df2, df3, df4, df5, df6, df7, df8, df9 = open_data()
    df_dirty = concat_data(df1, df2, df3, df4, df5, df6, df7, df8, df9)
    df_without_duplicates = drop_duplicates(df_dirty)
    df_preprocessed = fill_nans(df_without_duplicates)

    wallpaper = Image.open('data/wallpaper.jpeg')

    return df_dirty, df_without_duplicates, df_preprocessed, wallpaper


def render_page(df_dirty, df_without_duplicates, df_preprocessed, wallpaper):
    st.title('Задача предсказания отклика клиентов банка')
    st.image(wallpaper)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([':mag: О приложении', ':soap: Подготовка данных',
                                                  ':dart: Целевая переменная',
                                                  ':chart_with_upwards_trend: Однофакторный анализ',
                                                  ':bar_chart: Матрицы корреляций',
                                                  ':green_book: Статистические тесты'])

    with tab1:
        st.write('Добро пожаловать в наше веб-приложение, предоставляющее разведочный анализ данных.')
        st.write('В данном проекте мы решаем задачу предсказания отклика клиентов банка.')
        st.markdown("**Описание данных:**")

        data = {
            'Колонка': ['AGREEMENT_RK', 'AGE', 'GENDER', 'EDUCATION', 'MARITAL_STATUS', 'CHILD_TOTAL', 'DEPENDANTS',
                       'SOCSTATUS_WORK_FL',
                       'SOCSTATUS_PENS_FL', 'REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE',
                       'FL_PRESENCE_FL',
                       'OWN_AUTO', 'GEN_INDUSTRY', 'GEN_TITLE', 'JOB_DIR', 'WORK_TIME', 'FAMILY_INCOME',
                       'PERSONAL_INCOME', 'CREDIT',
                       'TERM', 'FST_PAYMENT', 'LOAD_NUM_TOTAL', 'LOAN_NUM_CLOSED', 'TARGET'],
            'Описание': ['уникальный идентификатор объекта в выборке', 'возраст клиента',
                         'пол клиента (1 — мужчина, 0 — женщина)', 'образование', 'семейное положение',
                         'количество детей клиента', 'количество иждивенцев клиента',
                         'социальный статус клиента относительно работы (1 — работает, 0 — не работает)',
                         'социальный статус клиента относительно пенсии (1 — пенсионер, 0 — не пенсионер)',
                         'область регистрации клиента', 'область фактического пребывания клиента',
                         'почтовый адрес области', 'наличие в собственности квартиры (1 — есть, 0 — нет)',
                         'количество автомобилей в собственности', 'отрасль работы клиента', 'должность',
                         'направление деятельности внутри компании', 'время работы на текущем месте (в месяцах)',
                         'семейный доход (несколько категорий)', 'личный доход клиента (в рублях)',
                         'сумма последнего кредита клиента (в рублях)', 'срок кредита',
                         'первоначальный взнос (в рублях)', 'количество ссуд клиента',
                         'количество погашенных ссуд клиента',
                         'отклик на маркетинговую кампанию (1 — отклик был зарегистрирован, 0 — отклика не было)'],
            'Тип данных': ['int64', 'int64', 'int64', 'object', 'object', 'int64', 'int64', 'int64', 'int64', 'object',
                      'object', 'object',
                      'int64', 'int64', 'object', 'object', 'object', 'float64', 'object', 'float64', 'float64',
                      'int64',
                      'float64', 'int64', 'int64', 'int64']
        }

        table = pd.DataFrame(data)
        st.table(table)

    with tab2:
        st.subheader('Очистка от дубликатов')
        st.write('Очистим исходные данные от дубликатов')

        st.caption('До очистки дубликатов:')
        duplicates_count = df_dirty.duplicated().sum()
        duplicates_df = pd.DataFrame({'Количество дубликатов': [duplicates_count], 'Размер датасета': [df_dirty.shape]})
        st.write(duplicates_df)

        st.caption('После очистки дубликатов: ')
        duplicates_counts = df_without_duplicates.duplicated().sum()
        duplicates_df1 = pd.DataFrame({'Количество дубликатов': [duplicates_counts],
                                       'Размер датасета': [df_without_duplicates.shape]})
        st.write(duplicates_df1)
        st.markdown("---")

        st.subheader('Работа с пропущенными значениями')
        st.write('Посмотрим на пропуски в данных')

        series = df_without_duplicates.isnull().mean() * 100
        filtered_series = series[series > 0]
        filtered_series_with_header = filtered_series.reset_index()
        filtered_series_with_header.columns = ['Название колонки', 'Количество пропусков (в процентах)']
        st.write(filtered_series_with_header)

        st.write('Стратегия заполнения пропусков: ')
        st.markdown("""
        *   `GEN_INDUSTRY` - категориальный столбец (заполним самым часто встречающимся значением)
        *   `GEN_TITLE` - категориальный столбец (заполним самым часто встречающимся значением)
        *   `JOB_DIR` - категориальный столбец (заполним самым часто встречающимся значением)
        *   `WORK_TIME` - числовой столбец c выбросами (заполним медианой)
        """)

        st.caption('Количество пропусков до заполнения: ')

        null_counts = df_without_duplicates.isnull().sum()
        total_null_count = null_counts.sum()
        st.write(total_null_count)

        st.caption('Количество пропусков после заполнения: ')

        null_counts = df_preprocessed.isnull().sum()
        total_null_count = null_counts.sum()
        st.write(total_null_count)

        st.markdown("---")
        st.subheader('Характеристики столбцов')
        st.caption('Характеристики числовых столбцов')
        describenum = df_preprocessed.describe()
        st.write(describenum)

        st.caption('Характеристики категориальных столбцов')
        describecat = df_preprocessed.describe(include='object')
        st.write(describecat)

    with tab3:
        st.subheader('Анализ целевой переменной')
        st.markdown("Целевая переменная в задаче: `TARGET`. Отклик на маркетинговую кампанию (1 — отклик был "
                    "зарегистрирован, 0 — отклика не было).")

        fig = go.Figure(go.Bar(x=df_preprocessed['TARGET'].value_counts().keys(),
                               y=df_preprocessed['TARGET'].value_counts().values))
        fig.update_layout(xaxis_title="Класс", yaxis_title="Количество",title_font_color='#222',
                          title_text='Распределение целевой переменной', xaxis_title_font_color='#222',
                          yaxis_title_font_color='#222')
        st.plotly_chart(fig)

        st.write('Видим, что классы целевой переменной не сбалансированы')

        ser = df_preprocessed['TARGET'].value_counts(normalize=True) * 100
        ser = ser.reset_index()
        ser.columns = ['Значение целевой переменной', 'Количество (в процентах)']
        st.write(ser)

        st.markdown("""
        *К чему приводит дисбаланс классов?*
        * При несбалансированных классах модели могут склоняться к тому классу, который имеет большее количество 
        объектов. В результате модель может показать низкую производительность на меньшем классе.
        * Если классы не сбалансированы, то модель может легко переобучаться на обучающих данных, демонстрируя 
        высокую точность на обучающих данных и низкую производительность на новых данных.
        * Несбалансированная целевая переменная может привести к снижению общей точности модели.
        """)

    with tab4:
        st.subheader('Однофакторный анализ')
        st.caption('Проведем однофакторный анализ четырех признаков')
        st.markdown("#### Возраст клиента")
        st.write('В датасете содержится информация о клиентах в возрасте 21-67 лет. Данные представлены целыми числами')

        fig = go.Figure(go.Bar(x=df_preprocessed['AGE'].value_counts().keys(),
                               y=df_preprocessed['AGE'].value_counts().values))
        fig.update_layout(xaxis_title="Возраст", yaxis_title="Количество", xaxis_title_font_color='#222',
                          yaxis_title_font_color='#222')
        st.plotly_chart(fig)

        st.markdown("**Влияние на целевую переменную**")

        col1, col2 = st.columns(2)
        fig = sns.kdeplot(df_preprocessed['AGE'])
        col1.pyplot()
        col2.write('Из распределения переменной мы видим, что большинство клиентов банка находятся '
                   'в возрасте 25-40 лет.')

        col1, col2 = st.columns(2)
        plt.figure(figsize=(12, 6))
        fig = sns.countplot(x='AGE', hue='TARGET', data=df_preprocessed, width = 1.8)
        col2.pyplot()
        col2.write("""Из кросс-таблицы и графика зависимости переменной и таргета можно сделать вывод, что люди старше
         55 лет почти никогда не откликались на предложения банка. Лучше всего откликались 42-летние и 25-летние люди.
         Тяжело сделать вывод о какой-то явной зависимости между возрастом и откликом клиента""")

        table = pd.crosstab(df_preprocessed['AGE'], df_preprocessed['TARGET'], normalize='index') * 100
        col1.write(table)
        st.markdown("---")

        st.markdown("#### Пол клиента")
        st.write('GENDER — пол клиента (1 — мужчина, 0 — женщина). Переменная по факту является категориальной, '
                 'но выражается в целых числах.')

        fig = go.Figure(go.Bar(x=df_preprocessed['GENDER'].value_counts().keys(),
                               y=df_preprocessed['GENDER'].value_counts().values))
        fig.update_layout(xaxis_title="Пол", yaxis_title="Количество", xaxis_title_font_color='#222',
                          yaxis_title_font_color='#222')
        st.plotly_chart(fig)

        st.write('Мы видим, что эти классы не сбалансированы. Мужчин в датасете больше.')

        ser = df_preprocessed['GENDER'].value_counts(normalize=True) * 100
        ser = ser.reset_index()
        ser.columns = ['Значение целевой переменной', 'Количество (в процентах)']
        st.write(ser)

        st.markdown("**Влияние на целевую переменную**")

        col1, col2 = st.columns(2)
        plt.figure(figsize=(12, 6))
        fig = sns.countplot(x='GENDER', hue='TARGET', data=df_preprocessed, width = 0.8)
        col2.pyplot()
        col2.write("""Из графика зависимости и кросс-таблицы мы видим, что женщины охотнее откликаются на банковские 
        предложения, чем мужчины (не намного, на 2%). Тяжело сделать вывод о какой-то явной зависимости между полом и 
        откликом клиента""")

        table = pd.crosstab(df_preprocessed['GENDER'], df_preprocessed['TARGET'], normalize='index') * 100
        col1.write(table)

        st.markdown("---")

        st.markdown("#### Социальный статус относительно работы")
        st.write(
            'SOCSTATUS_WORK_FL — социальный статус клиента относительно работы (1 — работает, 0 — не работает). '
            'Переменная по факту является категориальной, но выражается в целых числах')

        fig = go.Figure(go.Bar(x=df_preprocessed['SOCSTATUS_WORK_FL'].value_counts().keys(),
                               y=df_preprocessed['SOCSTATUS_WORK_FL'].value_counts().values))
        fig.update_layout(xaxis_title="Наличие работы", yaxis_title="Количество", xaxis_title_font_color='#222',
                          yaxis_title_font_color='#222')
        st.plotly_chart(fig)

        st.write('Мы видим, что эти классы очень не сбалансированы. Работающих людей в датасете больше, что логично.')

        ser = df_preprocessed['SOCSTATUS_WORK_FL'].value_counts(normalize=True) * 100
        ser = ser.reset_index()
        ser.columns = ['Значение целевой переменной', 'Количество (в процентах)']
        st.write(ser)

        st.markdown("**Влияние на целевую переменную**")

        col1, col2 = st.columns(2)
        plt.figure(figsize=(12, 6))
        fig = sns.countplot(x='SOCSTATUS_WORK_FL', hue='TARGET', data=df_preprocessed, width=0.8)
        col2.pyplot()
        col2.write("""Из графика зависимости и кросс-таблицы мы видим, что работающие люди охотнее откликаются на 
        предложения банка. Мы можем сделать вывод, что существует некоторая зависимость между работой клиента и его 
        склонности к отклику""")

        table = pd.crosstab(df_preprocessed['SOCSTATUS_WORK_FL'], df_preprocessed['TARGET'], normalize='index') * 100
        col1.write(table)

        st.markdown("---")

        st.markdown("#### Личный доход клиента")
        st.write('Заметим, что в датасете содержится информация о клиентах с доходом 24 - 250 000 рублей. Данные '
                 'представлены вещественными числами.')

        fig = go.Figure(go.Bar(x=df_preprocessed['PERSONAL_INCOME'].value_counts().keys(),
                               y=df_preprocessed['PERSONAL_INCOME'].value_counts().values))
        fig.update_layout(xaxis_title="Доход", yaxis_title="Количество", xaxis_title_font_color='#222',
                          yaxis_title_font_color='#222')
        st.plotly_chart(fig)

        col1, col2 = st.columns(2)
        fig = sns.kdeplot(df_preprocessed['PERSONAL_INCOME'])
        col1.pyplot()
        col2.write(
            'Из распределения переменной мы видим, что распределение личного дохода до 50 000 рублей близко к '
            'нормальному, но после этой суммы график имеет хвост')

        col1, col2 = st.columns(2)
        filtered_data = df_preprocessed[df_preprocessed['PERSONAL_INCOME'] < 50000]
        fig = sns.kdeplot(filtered_data['PERSONAL_INCOME'])
        col1.pyplot()
        col2.write('Видим, что наше предположение о нормальности неверно. Но мы были правы, когда говорили о хвосте в '
                   'графике.')

        st.markdown("**Влияние на целевую переменную**")

        col1, col2 = st.columns(2)
        fig = plt.scatter(df_preprocessed['PERSONAL_INCOME'], df_preprocessed['TARGET'])
        col1.pyplot()
        col2.write('Из графика зависимости переменной и таргета можно сделать вывод, что люди с доходом больше 50 000 '
                   'хуже откликаются на предложения банка. Тяжело сделать вывод о какой-то явной зависимости между '
                   'возрастом и откликом клиента')

    with tab5:
        st.subheader('Матрицы корреляций')
        st.caption('Корреляция Пирсона (чувствительна к выбросам)')

        corr = df_preprocessed.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(15, 15), dpi=70)
        sns.heatmap(corr, cmap="Blues", annot=True, ax=ax)
        st.pyplot()

        st.caption('Корреляция Спирмана (более устойчива к выбросам)')

        corr = df_preprocessed.corr(numeric_only=True, method='spearman')
        fig, ax = plt.subplots(figsize=(15, 15), dpi=70)
        sns.heatmap(corr, cmap="Blues", annot=True, ax=ax)
        st.pyplot()

        st.markdown("""
        На данной тепловой карте мы видим высокую корреляцию между `LOAN_NUM_TOTAL` и `LOAD_NUM_CLOSED` (0.73), 
        между `CREDIT` и `FST_PAYMENT` (0.52), между `CREDIT` и `TERM` (0.52), между `SOCSTATUS_PENS_FL` и `AGE` (0.52), 
        между `DEPENDANTS` и `CHILD_TOTAL` (0.52)
        """)

    with tab6:
        st.subheader('χ²')
        st.caption("Хи-квадрат (χ²) - это статистический тест, который используется для определения наличия или "
                   "отсутствия статистически значимой связи между двумя или более категориальными переменными.")

        st.markdown("**`TARGET` и `SOCSTATUS_WORK_FL`**")

        cont_table = pd.crosstab(df_preprocessed['TARGET'], df_preprocessed['SOCSTATUS_WORK_FL'])
        chi2, p_val, dof, expected = stats.chi2_contingency(cont_table)
        table = pd.DataFrame({
            "Хи-квадрат": [chi2],
            "p-значение": [p_val],
            "Степени свободы": [dof],
            "Ожидаемые частоты": [expected.flatten().tolist()]
        })
        st.write(table)
        st.write("""
        P-значение является мерой того, насколько вероятно получить такое или более экстремальное значение статистики, 
        при условии, что нулевая гипотеза верна. В данном случае, очень маленькое p-значение (6.173185236607358e-22) 
        говорит о том, что существует очень маленькая вероятность получить такую же или более экстремальную статистику, 
        если нулевая гипотеза о независимости переменных верна. Это позволяет нам отвергнуть нулевую гипотезу и сделать 
        вывод о наличии статистически значимой зависимости между категориальными переменными.
        """)

        st.markdown("**`TARGET` и `GENDER`**")

        cont_table = pd.crosstab(df_preprocessed['TARGET'], df_preprocessed['GENDER'])
        chi2, p_val, dof, expected = stats.chi2_contingency(cont_table)
        table = pd.DataFrame({
            "Хи-квадрат": [chi2],
            "p-значение": [p_val],
            "Степени свободы": [dof],
            "Ожидаемые частоты": [expected.flatten().tolist()]
        })
        st.write(table)
        st.write("""В данном случае, очень низкое p-значение указывает на то, что нулевая гипотеза может быть 
        отвергнута, и есть статистически значимые доказательства в пользу существования взаимосвязи между переменными.
        """)

        st.markdown("**`TARGET` и `SOCSTATUS_PENS_FL`**")

        cont_table = pd.crosstab(df_preprocessed['TARGET'], df_preprocessed['SOCSTATUS_PENS_FL'])
        chi2, p_val, dof, expected = stats.chi2_contingency(cont_table)
        table = pd.DataFrame({
            "Хи-квадрат": [chi2],
            "p-значение": [p_val],
            "Степени свободы": [dof],
            "Ожидаемые частоты": [expected.flatten().tolist()]
        })
        st.write(table)
        st.write("""Малое p-значение (в данном случае очень близкое к нулю) указывает на то, что отличия между 
        наблюдаемыми и ожидаемыми частотами статистически значимы.""")

        st.markdown("**`TARGET` и `FL_PRESENCE_FL`**")

        cont_table = pd.crosstab(df_preprocessed['TARGET'], df_preprocessed['FL_PRESENCE_FL'])
        chi2, p_val, dof, expected = stats.chi2_contingency(cont_table)
        table = pd.DataFrame({
            "Хи-квадрат": [chi2],
            "p-значение": [p_val],
            "Степени свободы": [dof],
            "Ожидаемые частоты": [expected.flatten().tolist()]
        })
        st.write(table)
        st.write("""Эти результаты говорят о том, что существует некоторая связь между переменными или группами, которые
         были анализированы. Однако, p-значение в данном случае превышает стандартный уровень значимости 0.05, что 
         означает, что различия между группами не являются статистически значимыми.""")

        st.subheader('Тест Манна-Уитни')
        st.caption("Тест Манна-Уитни (или U-тест) - это непараметрический статистический тест, который используется "
                   "для сравнения средних значений двух независимых выборок. Он широко применяется в случаях, когда "
                   "данные не соответствуют требованиям для применения параметрического теста, например, когда данные "
                   "имеют не нормальное распределение или содержат выбросы.")

        st.markdown("**`TARGET` и `FST_PAYMENT`**")

        statistic, p_value = mannwhitneyu(df_preprocessed[df_preprocessed['TARGET'] == 0]['FST_PAYMENT'],
                                          df_preprocessed[df_preprocessed['TARGET'] == 1]['FST_PAYMENT'])

        table = pd.DataFrame({
            "Статистика": [statistic],
            "p-значение": [p_value],
        })
        st.write(table)

        st.write("""В данном случае, p-значение указывает на то, что существует статистически значимая связь или 
        различие между наблюдаемыми данными.""")

        st.markdown("**`TARGET` и `PERSONAL_INCOME`**")

        statistic, p_value = mannwhitneyu(df_preprocessed[df_preprocessed['TARGET'] == 0]['PERSONAL_INCOME'],
                                          df_preprocessed[df_preprocessed['TARGET'] == 1]['PERSONAL_INCOME'])

        table = pd.DataFrame({
            "Статистика": [statistic],
            "p-значение": [p_value],
        })
        st.write(table)

        st.write("""В данном случае, p-значение указывает на то, что существует статистически значимая связь или 
        различие между наблюдаемыми данными.""")


def load_page():
    df_dirty, df_without_duplicates, df_preprocessed, wallpaper = preload_content()

    st.set_page_config(layout="centered",
                       page_title="Отклики клиентов банка",
                       page_icon=':bank:')

    render_page(df_dirty, df_without_duplicates, df_preprocessed, wallpaper)


if __name__ == "__main__":
    load_page()