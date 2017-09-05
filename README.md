# README #

###Описание проекта###

Разработка обертки для MARS-camera, позволяющей имплементировать её в C++ программу. 

###Должны быть реализованы следующие функции:###

+ *loadConfig("path_to_file")* _done_partially (no custom marsCamera config)
> загружает в камеру настройки из файла

+ *getChipsNumber()* _done_
> количество чипов в камере

+ *setThresholds(int chip_id, int numver_of_thresholds, int * thresholds)* _done_
> задает пороги конкретному чипу

+ *getHeight()* _done_
> высота склеенного изображения из всех трех чипов

+ *getWidth()* _done_
> ширина склеенного изображения из всех трех чипов

+ *acquire(float expose_time, int * data)* _done_
> возвращает склеенное изображение из всех имеющихся чипов

+ *setBias(int bias)* _done_
> задает напряжение напряжение на чипах

+ *saveAsTxt(string image_name, string folder)* _done_
> сохраняет картинку в текстовый файл

###Использование текущей версии###

To compile code with verbose output use: make verb=1

Running code: ./code

To omit control printing change pyEmbedding:19: _pyEnv::pyEnv(): verbose(false)_

