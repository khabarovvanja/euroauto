# Система распознавания данных по фото документов

## Установка:

1. Клонируем репозиторий GIT:
```
    git clone https://github.com/nosovdmitry007/pasport_class.git -b YOLO_5
```
2. Устанавливаем необхоимые библиотеки:
```
    pip install --upgrade pip
    cd passport
    pip install -r requirements.txt
```
3. Необходимо загрузить [веса](https://disk.yandex.ru/d/phAGUf4b2XIEsw) моделей в папку `yolo5`
```
├── yolo5 
│   ├── about.txt
│   ├── detect.pt
│   ├── fio_INN.pt
│   ├── inn_rotation.pt
│   └── povorot7.pt
```
## Пример распознавания паспорта:
1. Создаём экземпляр класса:
```
   from passport_class import Passport
   psprt = Passport()
```
2. Запускаем распознавание данных:
```
   result = psprt.detect_passport('path_to_image.jpg')
```

На выходе получаем данные в виде словаря и флага.

```
result = ({'date_of_birth':'Дата рождения',
      'date_of_issue':'Дата выдачи',
      'first_name':'Имя',
      'gender':'Пол',
      'issued_by_whom':'Кем выдан',
      'patronymic':'Отчество',
      'place_of_birth':'Место рождения',
      'series_and_number':'Серия и номер',
      'surname':'Фамилия',
      'unit_code':'Код подразделения'},
      0)
```
Флаг = 1, паспорт не обнаружен, или хотя бы 1 поле не распознано

Флаг = 0, все поля паспорта распознаны 

## Пример распознавания ИНН:
1. Создаём экземпляр класса:
```
   from passport_class import INN
   innsprt = INN()
```
2. Запускаем распознавание данных:
```
   result = innsprt.detect_inn('path_to_image.jpg')
```

На выходе получаем данные в виде словаря и флага.

```
result = ({'fio': 'ИВАНОВ ИВАН ИВАНОВИЧ', 
            'inn': '482608013231'},
             0)
```
Флаг = 1, паспорт не обнаружен, или хотя бы 1 поле не распознано

Флаг = 0, все поля паспорта распознаны 


## Пример распознавания СНИЛС:
1. Создаём экземпляр класса:
```
   from passport_class import Snils
    snilssprt = Snils()
```
2. Запускаем распознавание данных:
```
   result = snilssprt.detect_inn('path_to_image.jpg')
```

На выходе получаем данные в виде словаря и флага.

```
result = ({'fio': 'ИВАНОВ ИВАН ИВАНОВИЧ', 
            'number_strah': '187-220-276 69'},
             0)
```
Флаг = 1, паспорт не обнаружен, или хотя бы 1 поле не распознано

Флаг = 0, все поля паспорта распознаны 
