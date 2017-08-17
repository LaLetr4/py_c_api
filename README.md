# README #

###Описание проекта###

Разработка обертки для MARS-camera, позволяющей имплементировать её в C++ программу. 

###Должны быть реализованы следующие функции:###

+ *loadConfig("path_to_file")* 
> загружает в камеру настройки из файла

+ *getChipsNumber()* 
> количество чипов в камере

+ *setThresholds(int chip_id, int numver_of_thresholds, int * thresholds)* 
> задает пороги конкретному чипу

+ *getHeight()* 
> высота склеенного изображения из всех трех чипов

+ *getWidth()* 
> ширина склеенного изображения из всех трех чипов

+ *acquire(float expose_time, int * data)* 
> возвращает склеенное изображение из всех имеющихся чипов

###Использование текущей версии###

To compile code with verbose output use: make verb=1
Running code: ./code

