# Система распознавания данных по фото документов

## Установка:

1. Клонируем репозиторий GIT:
```
    git clone https://github.com/khabarovvanja/euroauto.git -b YOLO_5
```
2. Устанавливаем необхоимые библиотеки:
```
    pip install --upgrade pip
    cd euroauto
    pip install -r requirements.txt
```
3. Необходимо загрузить [веса](https://drive.google.com/drive/folders/1-HT4W6z3k4mcdnG11pe2qxkYUSOwWBou?usp=sharing) моделей в папку `yolo5`
```
├── yolo5 
│   ├── about.txt
│   ├── sts_detect.pt
│   ├── sts_povorot.pt
│   └──
```
## Пример распознавания данных СТС:
1. Создаём экземпляр класса:
```
   from euroauto_class import Sts
   sts = Sts()
```
2. Запускаем распознавание данных:
```
   result = sts.predict('path_to_image.jpg')
```

На выходе получаем данные в виде словаря и флага.

```
result, flag = ({'sign':'XXXXXXXXX', 'vin':'XXXXXXXXXXXXXXXXX'}, 0)
```
flag = 1, СТС не обнаружено, или хотя бы 1 поле не распознано

flag = 0, все поля СТС распознаны 
