# SHARE_CV_HW
Репозиторий содержит решения задач CV из курса SHARE

В директории CIFAR100_research находится презентация исследования задачи классификации датасета CIFAR100 при ограничении числа обучаемых параметров модели.
В ноутбуке research1 содержится код модели с количеством параметров менее 1000 и параметрической модели для ограничения в 50k(nc=24), 500k(nc=248), 100k(nc=48), 10k(nc=4), 1000k(nc=496) и ее обучение.
Параметрическая модель использует идеи ResNet c BottleNeck блоками для экономии параметров. Также реализованы CutMix и CutOut для повышения качества модели.

В директории CompetitionCV находится презентация решения задачи классификации датасета пиццы занявшего 2 место в соревновании и ноутбуки с моделями и их обучением используемых в двух лучших ансамблях.
Используемые модели для решения задачи:regnet_y_3_2gf, resnext50_32x4d, efficientnet_b3.

В директории nnframework реализован нейросетевой фреймворк со слоями DenseLayer, ReLU, Softmax, FlattenLayer, MaxPooling, Conv2DLayer, Conv2DTrLayer(транспонированная свертка). Тесты написаны с помощью Keras.
